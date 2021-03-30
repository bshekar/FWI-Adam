#!/usr/bin/env python
# coding: utf-8

# # FWI with standard LBFGS, section 4.1 of Kuldeep & Shekar (2021)
# 
#
param = {'t0': 0.,
        'tn': 4000.,              # Simulation last 1 second (1000 ms)
        'f0': 0.010,              # Source peak frequency is 10Hz (0.010 kHz)
        'nshots': 32,            # Number of shots to create gradient from
        'maxiter': 64,            # Maximum number of iterations
        'maxfun': 64,              # Maximum number of function evaluations
        'nbpml': 20}              # nbpml thickness.

import numpy as np
import scipy
from devito import configuration
configuration['log-level'] = 'WARNING'
from distributed import Client, LocalCluster, wait
from dask_jobqueue import SLURMCluster
import cloudpickle as pickle
import time as time
import h5py
from scipy.signal import butter, lfilter
import os

#NBVAL_IGNORE_OUTPUT
from examples.seismic import demo_model, Model, plot_velocity, plot_perturbation, Receiver
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from devito import Function, clear_cache
from scipy import ndimage
import matplotlib.pyplot as plt

def get_true_model():
    ''' True model: hard Marmousi model
        '''
    model_marm = demo_model('marmousi2d-isotropic', data_path=os.path.join(os.path.expanduser('~') ,'devito/data')
    
    spacing = (15, 15)
    origin = (0., 0.)
    nbpml=20
    
    vp=model_marm.vp.data[nbpml:-nbpml, nbpml:-nbpml]
    vp0=scipy.ndimage.zoom(vp, 0.5)
    param['spacing'] = spacing
    vp0=vp0[vp0 != 1.5].reshape([500, 189])
    true_model = Model(vp=vp0, origin=origin, shape=vp0.shape, spacing=spacing, space_order=4, nbpml=nbpml)
    
    param['nbpml'] = nbpml
    return true_model

def get_initial_model():
    '''The initial guess for the subsurface model. Smoothed hard Marmousi model (inverse crime!)
        '''
    # Make sure both model are on the same grid
    model0 = get_true_model()
    mask = np.ma.masked_where(model0.vp.data==1.5, model0.vp.data)  ## Define mask so that water layer is not updated
    slowP = 1/model0.vp.data
    g = scipy.ndimage.gaussian_filter(np.array(slowP), sigma=20, order=0, mode='nearest')
    smoothVP=1/g
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
    """
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
    """
    residual.data[:] = d.data[:residual.shape[0], :] - obsdata
    f = .5*np.linalg.norm(residual.data.flatten())**2
                        
    # Compute gradient using the adjoint-state method. Note, this
    # backpropagates the data misfit through the model.
    grad = Function(name="grad", grid=model0.grid)
    solver.gradient(rec=residual, u=u0, grad=grad, checkpointing=False)
    g = np.array(grad.data[:])
    
    return fg_pair(f, g)


# In[17]:

def gradient(vp):
    
    nbpml = param['nbpml']
    shape = param['shape']
    
    fil = h5py.File('curr_vel.hdf5', 'w')
    shape = param['shape']
    vel_data = fil.create_dataset('velocity', (shape[0],shape[1]), dtype='f16')
    vel_data[:] = vp.astype(np.float16).reshape(shape[0],shape[1])
    fil.close()
    
    start_time = time.time()
  
    # Call Dask cluster
    cluster = SLURMCluster(cores=1, dashboard_address=8787, n_workers=25, memory="5 GB", walltime="01:00:00",  queue="CLUSTER", death_timeout=600,local_directory="dask-worker-space")
    client = Client(cluster)
    time.sleep(4)
    print(client)
    # Define work list
    work = [dict(param) for i in range(param['nshots'])]
    for i in  range(param['nshots']):
        work[i]['shot_id'] = i
    
    # Distribute worklist to workers.
    fgi = client.map(gradient_i, work, retries=1)

    fg = client.submit(sum, fgi).result()
    obj = fg.f
    #client.restart()
    client.retire_workers()
    client.close()
    cluster.close()
    time.sleep(4)

#    spacing = param['spacing']
#    f0 = param['f0']
#    smoothrad = np.round(0.8*np.amax(model0.vp.data)/(f0*spacing[0])).astype(int)
#    g = scipy.ndimage.gaussian_filter(fg.g[:].astype(np.float32), sigma=smoothrad, order=0, mode='nearest')
    g = fg.g[:]
    
    gradv = -g[nbpml:-nbpml,nbpml:-nbpml].flatten().astype(np.float64)
    
    loss.append(obj)

    return obj, gradv

from scipy import optimize

# Define bounding box constraints on the solution.
def apply_box_constraint(vp):
    # Maximum possible 'realistic' velocity is 5.0 km/sec
    # Minimum possible 'realistic' velocity is 1.4 km/sec
    return np.clip(vp, 1.4, 5.0)

# Many optimization methods in scipy.optimize.minimize accept a callback
# function that can operate on the solution after every iteration.

def fwi_callbacks(x):
    # Apply boundary constraint
    x.data[:] = apply_box_constraint(x)


def fwi(vp, ftol=0.00000001, maxiter=param['maxiter'], maxfun=param['maxfun']):
    result = optimize.minimize(gradient,
                               vp,
                               #args=(param, ),
                               method='L-BFGS-B', jac=True,
                               callback=fwi_callbacks,
                               options={
                               'maxiter':maxiter,
                               'maxfun':maxfun,
                               'maxcor':10,
                               'disp':True})
    return result


if __name__ == "__main__":
    global loss
    loss=[]
    
    model0 = get_initial_model()
    nbpml = param['nbpml']
    shape = model0.vp.data[nbpml:-nbpml,nbpml:-nbpml].shape
    param['shape'] = shape
    st_time = time.time()
    vp0 = model0.vp.data[nbpml:-nbpml, nbpml:-nbpml].flatten().astype(np.float64)
    result = fwi(vp0)
    print('FWI took:',time.time()-st_time)
    
    vp=result.x.astype(np.float32).reshape(shape)
    
    fil = h5py.File('32shots_vel_100iter_sig20.hdf5', 'w')
    vel_data = fil.create_dataset('velocity', (vp.shape[0],vp.shape[1]), dtype='f16')
    vel_data[:] = vp.astype(np.float16)
    fil.close()

    loss_iter = np.asarray(loss)
    fil = h5py.File('32shots_loss_100iter_sig20.hdf5', 'w')
    loss_data = fil.create_dataset('loss', loss_iter.shape, dtype='f16')
    loss_data[:] = loss_iter
    fil.close()

 
