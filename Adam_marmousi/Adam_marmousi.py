#!/usr/bin/env python
# coding: utf-8

# # FWI with Adam optimizer. Implements section 3.3 of Kuldeep & Shekar (2021)
# 
#
param = {'t0': 0.,
        'tn': 4000.,              # Simulation last 1 second (1000 ms)
        'f0': 0.010,              # Source peak frequency is 10Hz (0.010 kHz)
        'nshots': 32,             # Number of shots to create gradient from
        'maxiter': 40,            # Maximum number of iterations
        'maxfun':60,              # Maximum number of function evaluations
        'nbpml': 20,              # nbpml thickness.
        'num_epochs': 64,          # Number of epochs
        'num_mini_batches': 8,    # Number of mini batches
        'mini_batch_size': 4}     # Size of a mini batch

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
#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()
tf.compat.v1.enable_eager_execution()
import os
from scipy.signal import butter, lfilter

#NBVAL_IGNORE_OUTPUT
from examples.seismic import demo_model, Model, plot_velocity, plot_perturbation, Receiver
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from devito import Function, clear_cache
from scipy import ndimage
import matplotlib.pyplot as plt


def get_true_model():
 
    model_marm = demo_model('marmousi2d-isotropic', data_path='/home/bharath/devito/data')
    
    spacing = (15, 15)
    origin = (0., 0.)
    nbpml=20
    
    vp=model_marm.vp.data[nbpml:-nbpml, nbpml:-nbpml]
    vp0=scipy.ndimage.zoom(vp, 0.5)
    param['spacing'] = spacing
    vp0=vp0[vp0 != 1.5].reshape([500, 189])
    true_model = Model(vp=vp0, origin=origin, shape=vp0.shape, spacing=spacing, space_order=4, nbpml=nbpml)
    param['origin'] = origin
    param['space_order'] = 4
    param['nbpml'] = nbpml
    return true_model

def get_initial_model():
    '''The initial guess for the subsurface model. Smoothed hard Marmousi model (inverse crime!)
        '''
    # Make sure both model are on the same grid
    model0 = get_true_model()
    nbpml = model0.nbl
    #vp0=scipy.ndimage.zoom(model0.vp.data[nbpml:-nbpml,nbpml:-nbpml], 0.5)
    vp0 = model0.vp.data
    slowP = 1/vp0
    g = scipy.ndimage.gaussian_filter(np.array(slowP), sigma=50, order=0, mode='nearest')
    smoothVP=1/g
    origin = (0., 0.)
    spacing = (15.0, 15.0)
    model0.vp = smoothVP
    return model0

def get_inter_model():
    
    fil = h5py.File('curr_vel.hdf5', 'r')
    vp = fil['velocity'][:]
    fil.close()
    spacing = (15, 15)
    origin = (0., 0.)
    nbpml=20
    model0 = Model(vp=vp, origin=origin, shape=vp.shape, spacing=spacing, space_order=4, nbpml=nbpml)
    return model0


def draw_shots(mini_batch_size, num_mini_batches, seed):
    np.random.seed(seed)
    np.testing.assert_equal(param['nshots'], num_mini_batches*mini_batch_size, err_msg='mini_batch_size times num_mini_batches does not equal total number of shots')
    batches=np.random.choice(param['nshots'],size=(num_mini_batches, mini_batch_size), replace=False)
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
    
    model0 = get_inter_model()
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
    
    nbpml = param['nbpml']
    shape = param['shape']
    #vp0 = np.clip(vp.numpy().astype(np.float16).reshape(shape[0],shape[1]),1.0,6.0)
    vp0 = vp.numpy().astype(np.float32).reshape(shape[0],shape[1])
    
    fil = h5py.File('curr_vel.hdf5', 'w')
    vel_data = fil.create_dataset('velocity', (shape[0],shape[1]))
    vel_data[:] = vp0
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

    gradv = -(2.0*g[nbpml:-nbpml,nbpml:-nbpml]/(np.power(vp0,3.0) + np.finfo(np.float32).eps )).flatten().astype(np.float64)

    def gradvp(dobj):
        return gradv * dobj, None
    
    return obj, gradvp



if __name__ == "__main__":
    
    global loss, batchloss, learn_rate
    loss=[]
    learn_rate=[]
    batchloss=[]

    global_step = tf.Variable(0,trainable=False)
    beta1 = 1.0 - 1.0/param['mini_batch_size']
    beta2 = beta1
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01, beta2=0.9) 
    
    model0 = get_initial_model()
    nbpml = param['nbpml']
    vp_init = model0.vp.data[nbpml:-nbpml,nbpml:-nbpml].flatten().astype(np.float64)
    vp = tf.Variable(vp_init, dtype=tf.float64)
    shape = model0.vp.data[nbpml:-nbpml,nbpml:-nbpml].shape
    param['shape'] = shape
   
    st_time = time.time()
    mem_per_core = 160/param['mini_batch_size']
    
    #cluster = SLURMCluster(cores=1, dashboard_address=8787, n_workers=param['mini_batch_size'], memory="%d GB"%mem_per_core, walltime="01:00:00",  queue="CLUSTER", death_timeout=600,local_directory="dask-worker-space")
    #client = Client(cluster)
    #time.sleep(1)
    #print(client)
     
    seed = 42
    for epoch in range(param['num_epochs']):
        shot_batches=draw_shots(param['mini_batch_size'], param['num_mini_batches'], seed)
        epoch_cost = 0.
        
        cluster = SLURMCluster(cores=1, dashboard_address=8787, n_workers=param['mini_batch_size'], memory="%d GB"%mem_per_core, walltime="01:00:00",  queue="CLUSTER", death_timeout=600,local_directory="dask-worker-space")
        client = Client(cluster)
        time.sleep(1)
        print(client)
         
 
        for batch_num in range(param['num_mini_batches']):
            
            with tf.GradientTape() as t:
                current_loss = gradient(vp,shot_batches[batch_num,:])
            grads = t.gradient(current_loss, vp)
            optimizer.apply_gradients(zip([grads],[vp]))
            epoch_cost += current_loss.numpy()/param['mini_batch_size']
            batchloss.append(current_loss.numpy()/param['mini_batch_size'])

        seed = seed + 1
        
        client.retire_workers()
        client.close()
        cluster.close()
        time.sleep(1)
        loss.append(epoch_cost)
        print(epoch_cost)
        if (epoch+1) % 4 == 0 or  epoch==0 :
        #if epoch > -1 :
           fil = h5py.File('10Hz_5pt0nois_vel_stoch%d.hdf5'%epoch, 'w')
           vel_data = fil.create_dataset('velocity', (shape[0], shape[1]), dtype='f16')
           vp0 = vp.numpy().astype(np.float16).reshape([shape[0], shape[1]])
           vel_data[:] = vp0
           fil.close()                 
    
    #client.retire_workers()
    #client.close()
    #cluster.close()
    #time.sleep(1)    

    loss_iter = np.asarray(loss)
    fil = h5py.File('loss.hdf5', 'w')
    loss_data = fil.create_dataset('loss', loss_iter.shape, dtype='f16')
    loss_data[:] = loss_iter
    fil.close()
           
    batchloss_arr = np.asarray(batchloss)
    fil = h5py.File('batchloss.hdf5','w')
    bloss_data = fil.create_dataset('loss', batchloss_arr.shape, dtype='f16')
    bloss_data[:] = batchloss_arr
    fil.close()
    print('Stochastic FWI took:',time.time()-st_time)



