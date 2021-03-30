#!/usr/bin/env python
# coding: utf-8

# # Generate clean data for marmousi model, for example figure 1 of Kuldeep & Shekar (2021).



import numpy as np
import scipy
from devito import configuration
configuration['log-level'] = 'WARNING'
from distributed import Client, LocalCluster, wait
from dask_jobqueue import SLURMCluster
import cloudpickle as pickle
import time
import os

# # True and smooth velocity models
# 
# Marmousi hard and smooth

# In[10]:


param = {'t0': 0.,
         'tn': 4000.,              # Simulation last 1 second (1000 ms)
         'f0': 0.025,              # Source peak frequency is 10Hz (0.010 kHz)
         'nshots': 32,             # Number of shots to create gradient from
         'm_bounds': (0.08, 0.25), # Set the min and max slowness
         'nbpml': 20}              # nbpml thickness.


# In[11]:


#NBVAL_IGNORE_OUTPUT
from examples.seismic import demo_model, Model, plot_velocity, plot_perturbation, Receiver
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
import scipy as scipy
from scipy import ndimage


def get_true_model():
    
    model_marm = demo_model('marmousi2d-isotropic', data_path=os.path.join(os.path.expanduser('~') ,'devito/data'))

    spacing = (15, 15)
    origin = (0., 0.)  
    nbpml=20
    vp=model_marm.vp.data[nbpml:-nbpml, nbpml:-nbpml]
    vp0=scipy.ndimage.zoom(vp, 0.5)
    vp0=vp0[vp0 != 1.5].reshape([500, 189])
    true_model = Model(vp=vp0, origin=origin, shape=vp0.shape, spacing=spacing, space_order=4, nbpml=nbpml)
    return true_model

# ## Acquisition geometry and data generation
# 
# Define shot and receiver coordinates and generate true data

def dump_shot_data(shot_id, rec, geometry):
    ''' Dump shot data to disk.
    '''
    pickle.dump({'rec':rec, 'geometry': geometry}, open('32shots/shot_%d.p'%shot_id, "wb"))
    
def model_shots_i(param):
    """ Inversion crime alert! Here the worker is creating the
        'observed' data using the real model. For a real case
        the worker would be reading seismic data from disk.
    """
    true_model = get_true_model()
    shot_id = param['shot_id']
    
    source_locs=param['source_coordinates']
    src_loc = source_locs[shot_id,:]

    # Geometry 
    geometry = AcquisitionGeometry(true_model, param['rec_coordinates'], src_loc,
                                   param['t0'], param['tn'], src_type='Ricker',
                                   f0=param['f0'])
    # Set up solver.
    solver = AcousticWaveSolver(true_model, geometry, space_order=4)

    # Generate synthetic receiver data from true model.
    true_d, _, _ = solver.forward(vp=true_model.vp)

    dump_shot_data(shot_id, true_d, geometry)

def model_shots(param):
    # Define work list
    work = [dict(param) for i in range(param['nshots'])]
    for i in  range(param['nshots']):
        work[i]['shot_id'] = i
        model_shots_i(work[i])
            
    # Map worklist to cluster
    futures = client.map(model_shots_i, work, retries=2)
    #wait(futures)


# In[13]:

if __name__ == "__main__":
    source_coordinates = np.empty((param['nshots'], 2), dtype=np.float32)
    source_coordinates[:, 1] = 30
    source_coordinates[:, 0] = np.linspace(15.,7500.,param['nshots'],endpoint=False)
    
    # Number of receiver locations per shot.
    rec_coordinates = np.empty((500, 2))
    rec_coordinates[:,0] = np.arange(0., 7500.,15)
    rec_coordinates[:,1] = 30
    
    param['source_coordinates'] = source_coordinates
    param['rec_coordinates'] = rec_coordinates
    
    start_time = time.time()
    # Start Dask cluster
    cluster = SLURMCluster(cores=1, n_workers=32, dashboard_address=8787, memory="2 GB",queue="CLUSTER", death_timeout=600, local_directory="dask-worker-space")
    client = Client(cluster)
    client
    # Generate shot data.
    model_shots(param)
    print('Time elapsed:', time.time()-start_time)
    client.retire_workers() 
    client.close()
    cluster.close()
    
