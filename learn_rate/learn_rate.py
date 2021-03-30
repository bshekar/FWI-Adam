#!/usr/bin/env python
# coding: utf-8

# # Plot the found learning rate. Reproduce figure 3 of Kuldeep & Shekar (2021).
# # Requires LR_Adam_fwi to be run for various mini-batch sizes first


import scipy
from scipy import ndimage
from pylab import rcParams
import h5py
import numpy as np
try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    mpl.rc('text', usetex=False)
    mpl.rc('font', size=24)
    mpl.rc('figure', figsize=(8, 6))
except:
    plt = None
    cm = None

sigma=1

fil = h5py.File('32lossAdam_4_10hz.hdf5', 'r')
loss4 = scipy.ndimage.gaussian_filter(fil['loss'][:].astype(np.float32), sigma=sigma, order=0, mode='nearest')
fil.close()
fil = h5py.File('32lossAdam_8_10hz.hdf5', 'r')
loss8 = scipy.ndimage.gaussian_filter(fil['loss'][:].astype(np.float32), sigma=sigma, order=0, mode='nearest')
fil.close()
fil = h5py.File('32lossAdam_16_10hz.hdf5', 'r')
loss16 = scipy.ndimage.gaussian_filter(fil['loss'][:].astype(np.float32), sigma=sigma, order=0, mode='nearest')
fil.close()
fil = h5py.File('32lossAdam_32_10hz.hdf5', 'r')
loss32 = scipy.ndimage.gaussian_filter(fil['loss'][:].astype(np.float32), sigma=sigma, order=0, mode='nearest')
fil.close()


lr = np.geomspace(5*1e-4, 1.0, 20)


fig, ax = plt.subplots(figsize=(8, 6))
"""
plt.loglog(lr, loss4, '--ro', markersize=4, label="4")
plt.loglog(lr, loss8, '--ko', markersize=4, label="8")
plt.loglog(lr, loss16, '--mo', markersize=4, label="16")
plt.loglog(lr, loss32, '--co', markersize=4, label="32")
plt.legend(loc="lower left")
"""
plt.loglog(lr, loss4, 'r', lw=4.0, ls="dotted", label="4")
plt.loglog(lr, loss8, 'k', lw=4.0, label="8")
plt.loglog(lr, loss16, 'g', lw=4.0, ls="dashdot", label="16")
plt.loglog(lr, loss32, 'm', lw=4.0, ls="dashed", label="32")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02),
          ncol=4, fontsize=18)

"""
plt.semilogx(lr, loss4, '--ro', markersize=4, label="4")
plt.semilogx(lr, loss8, '--ko', markersize=4, label="8")
plt.semilogx(lr, loss16, '--mo', markersize=4, label="16")
plt.semilogx(lr, loss32, '--co', markersize=4, label="32 (full batch)")
plt.legend(fontsize=18, loc="upper left")
"""
plt.xlim([np.min(lr), 0.5])
plt.ylim([3.8*10**6, 6*10**6])
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\mathcal{J}_{B}$')
plt.grid('True')

plt.tight_layout(pad=0.5)

figstring='lr_find.pdf'
fig.savefig(figstring, bbox_inches='tight')


#plt.show()
