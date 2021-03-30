#!/usr/bin/env python
# coding: utf-8

# # Learning rate finder for any given mini batch size
# # as described in section 3.2 of Kuldeep & Shekar (2021)

param = {'t0': 0.,
        'tn': 4000.,              # Simulation last 1 second (1000 ms)
        'f0': 0.010,              # Source peak frequency is 10Hz (0.010 kHz)
        'nshots': 32,             # Number of shots to create gradient from
        'nbpml': 20,              # nbpml thickness.
        'train_steps': 20,          # Number of epochs
        #'num_mini_batches': 10,    # Number of mini batches
        'mini_batch_size': 32}     # Size of a mini batch

import numpy as np
import scipy
from devito import configuration
configuration['log-level'] = 'WARNING'
from distributed import Client, LocalCluster, wait
from dask_jobqueue import SLURMCluster
import cloudpickle as pickle
import time
import h5py
import tensorflow as tf
import os
tf.compat.v1.enable_eager_execution()
from scipy.signal import butter, lfilter

from examples.seismic import demo_model, Model, plot_velocity, plot_perturbation, Receiver
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from devito import Function, clear_cache
from scipy import ndimage
import matplotlib.pyplot as plt

def get_true_model():
    ''' True model: hard Marmousi model
        '''
    model_marm = demo_model('marmousi2d-isotropic', data_path=os.path.join(os.path.expanduser('~') ,'/devito/data')
    
    spacing = (15, 15)
    origin = (0., 0.)
    nbpml=20
    
    vp=model_marm.vp.data[nbpml:-nbpml, nbpml:-nbpml]
    vp0=scipy.ndimage.zoom(vp, 0.5)
    param['spacing'] = spacing
    vp0=vp0[vp0 != 1.5].reshape([500, 189])
    true_model = Model(vp=vp0, origin=origin, shape=vp0.shape, spacing=spacing, space_order=4, nbpml=nbpml)
    param['shape'] = true_model.vp.shape
    return true_model

def get_initial_model():
    '''The initial guess for the subsurface model. Smoothed hard Marmousi model (inverse crime!)
        '''
    # Make sure both model are on the same grid
    model0 = get_true_model()
    slowP = 1/model0.vp.data
    g = scipy.ndimage.gaussian_filter(np.array(slowP), sigma=50, order=0, mode='nearest')
    smoothVP=1/g
    model0.vp = smoothVP
    return model0

def draw_shots(mini_batch_size, train_steps, seed):
    np.random.seed(seed)
    #np.testing.assert_equal(param['nshots'], num_mini_batches*mini_batch_size, err_msg='mini_batch_size times num_mini_batches does not equal total number of shots')
    batches=np.random.choice(param['nshots'],size=(train_steps, mini_batch_size), replace=True)
    return batches

# ## Acquisition geometry and data generation
# 
# Define shot and receiver coordinates and generate true data


def load_shot_data(shot_id, dt):
    ''' Load shot data from disk, resampling to the model time step.
    '''
    pkl = pickle.load(open("../marmousi_forward/32shots_10Hz/shot_%d.p"%shot_id, "rb"))
    return pkl['geometry'].resample(dt), pkl['rec'].resample(dt)


# ## FWI gradient operator

class fg_pair:
    def __init__(self, f, g):
        self.f = f
        self.g = g
    
    def __add__(self, other):
        f = self.f + other.f
        g = self.g + other.g
        
        return fg_pair(f, g)
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


# Create FWI gradient kernel for a single shot
def gradient_i(param):
    # Load the current model and the shot data for this worker.
    # Note, unlike the serial example the model is not passed in
    # as an argument. Broadcasting large datasets is considered
    # a programming anti-pattern and at the time of writing it
    # it only worked relaiably with Dask master. Therefore, the
    # the model is communicated via a file.
    
    fil = h5py.File('curr_velAdam.hdf5', 'r')
    vp = fil['velocity'][:]
    fil.close()
    
    model0 = get_initial_model()
    model0.vp=vp
    dt = model0.critical_dt
    
    geometry, rec = load_shot_data(param['shot_id'], dt)
    
    geometry.model = model0
    # Set up solver.
    solver = AcousticWaveSolver(model0, geometry, space_order=4)
    
    # Compute simulated data and full forward wavefield u0
    d, u0, _ = solver.forward(save=True)
    # Compute the data misfit (residual) and objective function
    residual = Receiver(name='rec', grid=model0.grid, time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    
    obsdata = rec.data[:residual.shape[0], :]
    ### Add bandlimited noise to otherwise clean data: Note: this is a hacky way of doing it as I can't figure out how to add noise to the rec object in devito!!
    np.random.seed(seed=param['shot_id'])
    nois=5.0*np.random.randn(obsdata.shape[0], obsdata.shape[1])
    fs = 1.0/dt
    nyq = 0.5 * fs
    low = 0.01
    high = 0.2
    b, a = butter(5, [low, high], btype='band')
    bnois = lfilter(b, a, nois)

    noisdata = obsdata + bnois

    residual.data[:] = d.data[:residual.shape[0], :] - noisdata
    f = .5*np.linalg.norm(residual.data.flatten())**2

    # Compute gradient using the adjoint-state method. Note, this
    # backpropagates the data misfit through the model.
    grad = Function(name="grad", grid=model0.grid)
    solver.gradient(rec=residual, u=u0, grad=grad, checkpointing=False)
    g = np.array(grad.data[:])

    return fg_pair(f, g)    
                        

# In[17]:
@tf.custom_gradient
def gradient(vp,shot_batch):
    
    fil = h5py.File('curr_velAdam.hdf5', 'w')
    shape = param['shape']
    vel_data = fil.create_dataset('velocity', (shape[0],shape[1]), dtype='f16')
    vel_data[:] = vp.numpy().astype(np.float16).reshape(shape[0],shape[1])
    fil.close()
    
    start_time = time.time()
    
    
    # Define work list --- work with mini batches!
    work = [dict(param) for i in range(param['mini_batch_size'])]
    for i in  range(param['mini_batch_size']):
        work[i]['shot_id'] = shot_batch[i]
    
    
    # Distribute worklist to workers.
    fgi = client.map(gradient_i, work, retries=1)

    fg = client.submit(sum, fgi).result()
    obj = fg.f
    

    g = fg.g[:]
    
    gradv = -(2.0*g.flatten()/(np.power(vp.numpy(),3))).astype(np.float64)
    
    
    
    def gradvp(dobj):
        return gradv * dobj, None
    
    return obj, gradvp



if __name__ == "__main__":
    
    global loss, learn_rate
    loss=[]
    learn_rate=[]

    model0 = get_initial_model()
    vp = tf.Variable(model0.vp.data.flatten().astype(np.float64), dtype=tf.float64)

    lr = 4.5*1e-4
    st_time = time.time()
    mem_per_core = 170/param['mini_batch_size'] 
    seed = 42
    shot_batches=draw_shots(param['mini_batch_size'], param['train_steps'], seed)
    cluster = SLURMCluster(cores=1, dashboard_address=8000, n_workers=param['mini_batch_size'], memory="%d GB"%mem_per_core, walltime="01:00:00",  queue="CLUSTER", death_timeout=600,local_directory="dask-worker-space")
    client = Client(cluster)
    time.sleep(4)
    print(client)
    learn_rate = np.geomspace(5*1e-4, 1.0, num=param['train_steps'])
    for train_step in range(param['train_steps']):
        lr = learn_rate[train_step]
    #    cluster = SLURMCluster(cores=1, dashboard_address=8000, n_workers=param['mini_batch_size'], memory="%d GB"%mem_per_core, walltime="01:00:00",  queue="CLUSTER", death_timeout=600,local_directory="dask-worker-space")
     #   client = Client(cluster)
        #time.sleep(4)
        with tf.GradientTape() as t:
            current_loss = gradient(vp,shot_batches[train_step,:])
        grads = t.gradient(current_loss, vp)
        tf.compat.v1.train.AdamOptimizer(learning_rate=lr).apply_gradients(zip([grads],[vp]))
        epoch_cost = current_loss.numpy()/param['mini_batch_size']
        loss.append(epoch_cost)
        print(epoch_cost)
        print(lr)
        #lr = lr*1.5    
     #   client.retire_workers()
     #   client.close()
     #   cluster.close()

    client.retire_workers()
    client.close()
    cluster.close()
    time.sleep(4)
    print('Stochastic FWI took:',time.time()-st_time)


    loss_iter = np.asarray(loss)
    fil = h5py.File('32lossAdam_32_10hz.hdf5', 'w')
    loss_data = fil.create_dataset('loss', loss_iter.shape, dtype='f16')
    loss_data[:] = loss_iter
    fil.close()
