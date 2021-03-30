#!/usr/bin/env python
# coding: utf-8

# # Plot figures, for example figures 7 and 8 of Kuldeep & Shekar (2021)
#
#

from scipy import ndimage
from devito import Function, clear_cache
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry
from examples.seismic import demo_model, Model, Receiver
import cloudpickle as pickle
import h5py
from devito import configuration
import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import rcParams
import os
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
#mpl.rc('text', usetex=True)
mpl.rc('font', size=24)
mpl.rc('figure', figsize=(8, 6))


param = {'t0': 0.,
         'tn': 4000.,              # Simulation last 1 second (1000 ms)
         'f0': 0.010,              # Source peak frequency is 10Hz (0.010 kHz)
         'm_bounds': (0.08, 0.25), # Set the min and max slowness
	     'nshots': 1,
         'nbpml': 20}              # nbpml thickness.


configuration['log-level'] = 'WARNING'

# NBVAL_IGNORE_OUTPUT


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
                       spacing=spacing, space_order=4, nbpml=nbpml)
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


def get_inverted_model(iter):
    '''The inverted model '''
    fil = h5py.File('10Hz_5pt0nois_vel_stoch'+str(iter-1)+'.hdf5', 'r')
    vp = fil['velocity'][:]
    fil.close()
    # Make sure both model are on the same grid
    model1 = get_true_model()
    #vp[:,0:40]=1.5
    #model1.vp = np.clip(vp, 1.2, 5.0)
    model1.vp = vp
    return model1


def read_shot_data(shot_id, nois_lvl):
    fil = h5py.File('shots/true_shot_%d.hdf5' % shot_id, 'r')
    true_d = fil['velocity'][:].astype(np.float64)
    fil.close()
    fil = h5py.File('shots/init100_shot_%d.hdf5' % shot_id, 'r')
    init100_d = fil['velocity'][:].astype(np.float64)
    fil.close()
    fil = h5py.File('shots/init50_shot_%d.hdf5' % shot_id, 'r')
    init50_d = fil['velocity'][:].astype(np.float64)
    fil.close()
    fil = h5py.File('shots/init10_shot_%d.hdf5' % shot_id, 'r')
    init10_d = fil['velocity'][:].astype(np.float64)
    fil.close()
    fil = h5py.File('shots/inv_shot_%d.hdf5' % shot_id, 'r')
    inv_d = fil['velocity'][:].astype(np.float64)
    fil.close()
    
    dt = get_true_model().critical_dt    
    ### Add noise to shot data
    #np.random.seed(seed=param['shot_id'])
    nois=nois_lvl*np.random.randn(true_d.shape[0], true_d.shape[1])
    fs = 1.0/dt
    nyq = 0.5 * fs
    low = 0.01
    high = 0.5
    b, a = butter(5, [low, high], btype='band')
    bnois = lfilter(b, a, nois)

    nois_d = true_d + bnois
    return true_d, nois_d, init100_d, init50_d, init10_d, inv_d

def plot_velocity(model, source=None, receiver=None, colorbar=True):

    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    field = (getattr(model, 'vp', None) or getattr(model, 'lam')).data[slices]
    #field = np.clip(field, 1.0, 5.0)
    vel = get_true_model().vp.data
    vmin = np.min(vel)
    vmax = np.max(vel)
    plot = plt.imshow(np.transpose(field), animated=True, cmap=cm.jet,
                      vmin=vmin, vmax=vmax,
                      extent=extent)
    plt.xlabel(r'$x$ (km)')
    plt.ylabel(r'$z$ (km)')
    #plt.tight_layout(pad=0.2)
    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(1e-3*receiver[:, 0], 1e-3*receiver[:, 1],
                    s=25, c='green', marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(1e-3*source[:, 0], 1e-3*source[:, 1],
                    s=25, c='red', marker='o')

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label(r'$V_{P}$ (km/s)')
#    plt.show()

def plot_update(model, vmin, vmax):
    
              domain_size = 1.e-3 * np.array(model.domain_size)
              extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]
        
              slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
              field = (getattr(model, 'vp', None) or getattr(model, 'lam')).data[slices]
              #field = np.clip(field, 1.0, 5.0)
              vel = get_true_model().vp.data

              plot = plt.imshow(np.transpose(field), animated=True, cmap=cm.jet,
                                vmin=vmin, vmax=vmax,
                                extent=extent)
              plt.xlabel(r'$x$ (km)')
              plt.ylabel(r'$z$ (km)')
              
        
              # Ensure axis limits
              plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
              plt.ylim(model.origin[1] + domain_size[1], model.origin[1])
                
                # Create aligned colorbar on the right
              colorbar=True
              if colorbar:
                ax = plt.gca()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(plot, cax=cax)
                cbar.set_label('$V_{P}$ (km/s)')

def plot_perturbation(model, model0, source=None, receiver=None, colorbar=True):

    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    field1 = (getattr(model, 'vp', None) or getattr(model, 'lam')).data[slices] 
    field2 = (getattr(model0, 'vp', None) or getattr(model, 'lam')).data[slices]
    #field = np.clip(field1 - field2, -2.0, 2.0)
    field = field1 - field2
    val=np.percentile(field, 99)
    val = 1.0
    plot = plt.imshow(np.transpose(field), animated=True, cmap=cm.jet,
                      vmin=-val, vmax=val,
                      extent=extent)
    plt.xlabel(r'$x$ (km)')
    plt.ylabel(r'$z$ (km)')
    #plt.tight_layout(pad=0.2)
    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(1e-3*receiver[:, 0], 1e-3*receiver[:, 1],
                    s=25, c='green', marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(1e-3*source[:, 0], 1e-3*source[:, 1],
                    s=25, c='red', marker='o')

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label(r'$V_{P}$ (km/s)')


def plot_shotrecord(rec, model, t0, tn, scale, colorbar=True):

    #scale = 0.1*np.max(rec)
    extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0],
              1e-3*tn, t0]

    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
    plt.xlabel(r'$x$ (km)')
    plt.ylabel(r'$t$ (s)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(plot, cax=cax)
        cbar.set_label(r'$P$', rotation=90)
#    plt.show()


def plot_logs(model0, model1, modelt, xcord):
    
    domain_size = 1.e-3 * np.array(model0.domain_size)
    extent = [model0.origin[0], model0.origin[0] + domain_size[0],
              model0.origin[1] + domain_size[1], model0.origin[1]]
    nbl = model0.nbl
    vp0 = model0.vp.data[nbl:-nbl, nbl:-nbl]
    vp1 = model1.vp.data[nbl:-nbl, nbl:-nbl]
    vpt = modelt.vp.data[nbl:-nbl, nbl:-nbl]

    z = 1.e-3 * np.arange(modelt.origin[1], modelt.shape[1]*modelt.spacing[1], modelt.spacing[1])
    plt.plot(vp0[xcord,:], z, 'k')
    plt.plot(vp1[xcord,:], z,  'r')
    plt.plot(vpt[xcord,:], z, 'b')
    plt.xlabel(r'$V_{P}$ (km/s)')
    plt.ylabel(r'$z$ (km)')
    plt.gca().invert_yaxis()
    plt.grid('True')



if __name__ == "__main__":
    print("generating plots, check hard coded defaults")
    #source_coordinates = np.empty((25, 2), dtype=np.float32)
    #source_coordinates[:, 1] = 30
    #source_coordinates[:, 0] = np.linspace(15.,7500.,25,endpoint=False)
    
    # Number of receiver locations per shot.
    #rec_coordinates = np.empty((500, 2))
    #rec_coordinates[:,0] = np.arange(0., 7500.,15)
    #rec_coordinates[:,1] = 30

iter=64
f0=10
nois_lvl=5.0

model10 = get_initial_model(10)
model50 = get_initial_model(50)
model100 = get_initial_model(100)
model1 = get_inverted_model(iter)
modelt = get_true_model()


fig, ax = plt.subplots()
plot_velocity(model1, colorbar=True)
figstring='figures/inv_model'+str(iter)+'.pdf'
fig.savefig(figstring, bbox_inches='tight')

fig, ax = plt.subplots()
plot_velocity(model10, colorbar=True)
figstring='figures/init_model10.pdf'
fig.savefig(figstring, bbox_inches='tight')

fig, ax = plt.subplots()
plot_velocity(model50, colorbar=True)
figstring='figures/init_model50.pdf'
fig.savefig(figstring, bbox_inches='tight')

fig, ax = plt.subplots()
plot_velocity(model100, colorbar=True)
figstring='figures/init_model100.pdf'
fig.savefig(figstring, bbox_inches='tight')

fig, ax = plt.subplots()
plot_perturbation(modelt, model1, colorbar=True)
figstring='figures/pert_model'+str(iter)+'.pdf'
fig.savefig(figstring, bbox_inches='tight')

for shotnum in range(1):
        shot_id = 0
        true_d, nois_d, init100_d, init50_d, init10_d, inv_d = read_shot_data(shot_id, nois_lvl)
        scale = 0.1*np.max(true_d)
        #scale=1.0
        fig, ax = plt.subplots()
        plot_shotrecord(true_d, modelt, 0, 4000, scale)
        fig.savefig('figures/true_%d.pdf'%shot_id, bbox_inches='tight')
        fig, ax = plt.subplots()
        plot_shotrecord(init50_d, modelt, 0, 4000, scale)
        fig.savefig('figures/init50_%d.pdf' % shot_id, bbox_inches='tight')
        fig, ax = plt.subplots()
        plot_shotrecord(true_d-init50_d, modelt, 0, 4000, scale)
        fig.savefig('figures/true_init50_%d.pdf' % shot_id, bbox_inches='tight')
        fig, ax = plt.subplots()
        plot_shotrecord(init10_d, modelt, 0, 4000, scale)
        fig.savefig('figures/init50_%d.pdf' % shot_id, bbox_inches='tight')
        fig, ax = plt.subplots()
        plot_shotrecord(true_d-init10_d, modelt, 0, 4000, scale)
        fig.savefig('figures/true_init50_%d.pdf' % shot_id, bbox_inches='tight')
        fig, ax = plt.subplots()
        plot_shotrecord(inv_d, modelt, 0, 4000, scale)
        fig.savefig('figures/inv_%d.pdf' % shot_id, bbox_inches='tight')
        fig, ax = plt.subplots()
        plot_shotrecord(nois_d, modelt, 0, 4000, scale)
        fig.savefig('figures/nois_%d.pdf'%shot_id, bbox_inches='tight')
        fig, ax = plt.subplots()
        plot_shotrecord(nois_d-inv_d, modelt, 0, 4000, scale)
        fig.savefig('figures/nois_inv_%d.pdf' % shot_id, bbox_inches='tight')


###  Illustrate cycle skipping, with 2D to 3D correction

time = np.linspace(0, 4000, true_d.shape[0])
dt = modelt.critical_dt

ind1=np.where(np.isclose(time, 2000, atol=dt))[0]
ind2=np.where(np.isclose(time, 2750, atol=dt))[0]

"""
epsilon = 10**(-8)
true_tr = np.convolve(np.sqrt(time/1000 +epsilon)*true_d[:,0], 1.0/np.sqrt(time/1000 +epsilon), mode='same')
init50_tr = np.convolve(np.sqrt(time/1000 +epsilon)*init50_d[:,0], 1.0/np.sqrt(time/1000 +epsilon), mode='same') 

fig, ax = plt.subplots()
plt.plot(time[ind1[0]:ind2[0]], true_tr[ind1[0]:ind2[0]], color='black', lw=2.0)
plt.plot(time[ind1[0]:ind2[0]], init50_tr[ind1[0]:ind2[0]], color='red', lw=2.0)
plt.ylabel(r'$P$ ')
plt.xlabel(r'$t$ (ms)')
fig.savefig('figures/far_off_data_sig50_'+str(f0)+'hz.pdf', bbox_inches='tight')
"""


fig, ax = plt.subplots()
plt.plot(time[ind1[0]:ind2[0]], true_d[ind1[0]:ind2[0],0], color='black', lw=2.0, ls='solid')
plt.plot(time[ind1[0]:ind2[0]], init50_d[ind1[0]:ind2[0],0], color='gray', lw=2.0, ls='dotted')
plt.plot(time[ind1[0]:ind2[0]], inv_d[ind1[0]:ind2[0],0], color='red', lw=2.0, ls='dashed')
plt.ylabel(r'$P$ ')
plt.xlabel(r'$t$ (ms)')
fig.savefig('figures/far_off_data_sig50_'+str(f0)+'hz.pdf', bbox_inches='tight')


fig, ax = plt.subplots()
plt.plot(time[ind1[0]:ind2[0]], true_d[ind1[0]:ind2[0],0], color='black', lw=2.0, ls='solid')
plt.plot(time[ind1[0]:ind2[0]], init10_d[ind1[0]:ind2[0],0], color='gray', lw=2.0, ls='dotted')
plt.ylabel(r'$P$ ')
plt.xlabel(r'$t$ (ms)')
fig.savefig('figures/far_off_data_sig10_'+str(f0)+'hz.pdf', bbox_inches='tight')


fig, ax = plt.subplots()
plt.plot(time[ind1[0]:ind2[0]], true_d[ind1[0]:ind2[0],0], color='black', lw=2.0, ls='solid')
plt.plot(time[ind1[0]:ind2[0]], init100_d[ind1[0]:ind2[0],0], color='gray', lw=2.0, ls='dotted')
plt.ylabel(r'$P$ ')
plt.xlabel(r'$t$ (ms)')
fig.savefig('figures/far_off_data_sig100_'+str(f0)+'hz.pdf', bbox_inches='tight')


true_model=get_true_model()
source_coordinates = np.empty((param['nshots'], 2), dtype=np.float32)
source_coordinates[:, 1] = 30
source_coordinates[:, 0] = 250 * 15

# Number of receiver locations per shot.
rec_coordinates = np.empty((500, 2))
rec_coordinates[:, 0] = np.arange(0., 7500., 15)
rec_coordinates[:, 1] = 150

geometry = AcquisitionGeometry(true_model, rec_coordinates, source_coordinates,
                               param['t0'], param['tn'], src_type='Ricker',
                               f0=param['f0'])

wav = geometry.src.data[:, 0]
time = np.arange(0, len(wav)) * dt
freq = np.fft.rfftfreq(len(wav), d=dt)
amp = np.abs(np.fft.rfft(wav))

fig, ax = plt.subplots()
plt.plot(time, wav, color='black', lw=2.0, ls='solid')
plt.xlim([0, 500])
plt.ylim([-0.5, 1.2])
plt.ylabel(r'$P$ ')
plt.xlabel(r'$t$ (ms)')
fig.savefig('figures/wav_' + str(f0) + 'hz.pdf', bbox_inches='tight')

fig, ax = plt.subplots()
plt.plot(freq*1000, amp/np.max(amp), color='black', lw=2.0, ls='solid')
plt.ylim([0.0, 1.2])
plt.xlim([0, 60])
plt.ylabel(r'$P$ ')
plt.xlabel(r'$f$ (Hz)')
fig.savefig('figures/wavspec_' + str(f0) + 'hz.pdf', bbox_inches='tight')
