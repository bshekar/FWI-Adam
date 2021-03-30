#!/usr/bin/env python
# coding: utf-8

# # Generate some data, for example figures 7c and d of Kuldeep & Shekar (2021)
# 
# 


import numpy as np
import scipy
from devito import configuration
configuration['log-level'] = 'WARNING'
from distributed import Client, LocalCluster, wait
import cloudpickle as pickle
import time
import h5py
import os


# # True and smooth velocity models
# 
# Marmousi hard and smooth

# In[10]:


param = {'t0': 0.,
         'tn': 4000.,              # Simulation last 1 second (1000 ms)
         'f0': 0.010,              # Source peak frequency is 10Hz (0.010 kHz)
         'm_bounds': (0.08, 0.25), # Set the min and max slowness
	     'nshots': 1,
         'nbpml': 20}              # nbpml thickness.


# In[11]:


#NBVAL_IGNORE_OUTPUT
from examples.seismic import demo_model, Model, plot_velocity, plot_perturbation, Receiver
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
import scipy as scipy
from scipy import ndimage


# ## Acquisition geometry and data generation
# 
# Define shot and receiver coordinates and generate true data

def get_true_model():
    ''' True model: hard Marmousi model
        '''
    model_marm = demo_model('marmousi2d-isotropic',data_path=os.path.join(os.path.expanduser('~') ,'devito/data'))

    spacing = (15, 15)
    origin = (0., 0.)
    nbpml = 20

    vp = model_marm.vp.data[nbpml:-nbpml, nbpml:-nbpml]
    vp0 = scipy.ndimage.zoom(vp, 0.5)
    vp0=vp0[vp0 != 1.5].reshape([500, 189])
    param['spacing'] = spacing
    true_model = Model(vp=vp0, origin=origin, shape=vp0.shape,
                       spacing=spacing, space_order=4, nbl=10*nbpml)
    param['shape'] = true_model.vp.shape
    return true_model


def get_initial_model(sigma):
    '''The initial guess for the subsurface model. Smoothed hard Marmousi model (inverse crime!)
        '''
    # Make sure both model are on the same grid
    model0 = get_true_model()
    # Define mask so that water layer is not updated
    slowP = 1/model0.vp.data
    g = scipy.ndimage.gaussian_filter(
        np.array(slowP), sigma=sigma, order=0, mode='nearest')
    smoothVP = 1/g
    model0.vp = smoothVP
    return model0


def model_shots_i(param):

    true_model = get_true_model()
    init_model100 = get_initial_model(100.0)
    init_model50 = get_initial_model(50.0)
    init_model10 = get_initial_model(10.0)
    inv_model = get_initial_model(50.0)

    iter=64
    fil = h5py.File('10Hz_5pt0nois_vel_stoch' + str(iter - 1) + '.hdf5', 'r')
    vp = fil['velocity'][:]
    fil.close()
    inv_model.vp = vp
    
    shot_id = param['shot_id']
    
    source_locs=param['source_coordinates']
    src_loc = source_locs[shot_id,:]

    # Geometry 
    geometry = AcquisitionGeometry(true_model, param['rec_coordinates'], src_loc,
                                   param['t0'], param['tn'], src_type='Ricker',
                                   f0=param['f0'])
    # Set up solver.
    solver = AcousticWaveSolver(true_model, geometry, space_order=4, fs=True)
    

    # Generate data from true, initial and true models.
    true_d, _, _ = solver.forward(vp=true_model.vp)
    init_d100, _, _ = solver.forward(vp=init_model100.vp)
    init_d50, _, _ = solver.forward(vp=init_model50.vp)
    init_d10, _, _ = solver.forward(vp=init_model10.vp)
    inv_d, _, _ = solver.forward(vp=inv_model.vp)
    
    dt = true_model.critical_dt
    true_data = true_d.resample(dt).data
    init_data100 = init_d100.resample(dt).data
    init_data50 = init_d50.resample(dt).data
    init_data10 = init_d10.resample(dt).data
    inv_data = inv_d.resample(dt).data

    # Dump data to h5py files

    fil = h5py.File('shots/true_shot_%d.hdf5'%shot_id, 'w')
    shot_data = fil.create_dataset('velocity', (true_data.shape[0], true_data.shape[1]), dtype='f16')
    shot_data[:] = true_data
    fil.close()

    fil = h5py.File('shots/init100_shot_%d.hdf5'%shot_id, 'w')
    shot_data = fil.create_dataset('velocity', (init_data100.shape[0], init_data100.shape[1]), dtype='f16')
    shot_data[:] = init_data100
    fil.close()

    fil = h5py.File('shots/init50_shot_%d.hdf5'%shot_id, 'w')
    shot_data = fil.create_dataset('velocity', (init_data50.shape[0], init_data50.shape[1]), dtype='f16')
    shot_data[:] = init_data50
    fil.close()

    fil = h5py.File('shots/init10_shot_%d.hdf5'%shot_id, 'w')
    shot_data = fil.create_dataset('velocity', (init_data10.shape[0], init_data10.shape[1]), dtype='f16')
    shot_data[:] = init_data10
    fil.close()

    fil = h5py.File('shots/inv_shot_%d.hdf5'%shot_id, 'w')
    shot_data = fil.create_dataset('velocity', (inv_data.shape[0], inv_data.shape[1]), dtype='f16')
    shot_data[:] = inv_data
    fil.close()




def model_shots(param):
    # Define work list
    shots = [0]
    work = [dict(param) for i in range(1)]
    for i in  range(1):
    	shot_id = shots[i]
    	work[i]['shot_id'] = shot_id
    	model_shots_i(work[i])
            
    # Map worklist to cluster
    futures = client.map(model_shots_i, work, retries=2)
    #wait(futures)


# In[13]:
#import sys
if __name__ == "__main__":
    source_coordinates = np.empty((param['nshots'], 2), dtype=np.float32)
    source_coordinates[:, 1] = 30
    source_coordinates[:, 0] = 250*15
    
    # Number of receiver locations per shot.
    rec_coordinates = np.empty((500, 2))
    rec_coordinates[:,0] = np.arange(0., 7500.,15)
    rec_coordinates[:,1] = 150
    
    param['source_coordinates'] = source_coordinates
    param['rec_coordinates'] = rec_coordinates
    
    start_time = time.time()
    # Start Dask cluster
    cluster = LocalCluster(n_workers=4, death_timeout=600)
    client = Client(cluster)
    
    # Generate shot data.
    model_shots(param)
    print('Time elapsed:', time.time()-start_time)

    client.retire_workers()
    client.close()
    cluster.close()
