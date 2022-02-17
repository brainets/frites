"""
Estimate comodulations between brain areas
==========================================

This example illustrates how to estimate the instantaneous comodulations using
mutual information between pairwise ROI and also perform statistics.
"""
import numpy as np
import xarray as xr

from itertools import product

from frites.simulations import sim_multi_suj_ephy
from frites.dataset import DatasetEphy
from frites.workflow import WfConnComod
from frites import set_mpl_style

import matplotlib.pyplot as plt
set_mpl_style()


###############################################################################
# Simulate electrophysiological data
# ----------------------------------
#
# Let's start by simulating MEG / EEG electrophysiological data coming from
# multiple subjects using the function
# :func:`frites.simulations.sim_multi_suj_ephy`. As a result, the `x` output
# is a list of length `n_subjects` of arrays, each one with a shape of
# n_epochs, n_sites, n_times

modality = 'meeg'
n_subjects = 5
n_epochs = 50
n_times = 100
x, roi, _ = sim_multi_suj_ephy(n_subjects=n_subjects, n_epochs=n_epochs,
                               n_times=n_times, modality=modality,
                               random_state=0, n_roi=4)
times = np.linspace(-1, 1, n_times)

###############################################################################
# Simulate spatial correlations
# -----------------------------
#
# Bellow, we start by simulating some distant correlations by injecting the
# activity of an ROI to another
for k in range(n_subjects):
    x[k][:, [1], slice(20, 40)] += x[k][:, [0], slice(20, 40)]
    x[k][:, [2], slice(60, 80)] += x[k][:, [3], slice(60, 80)]
print(f'Corr 1 : {roi[0][0]}-{roi[0][1]} between [{times[20]}-{times[40]}]')
print(f'Corr 2 : {roi[0][2]}-{roi[0][3]} between [{times[60]}-{times[80]}]')

###############################################################################
# Define the electrophysiological dataset
# ---------------------------------------
#
# Now we define an instance of :class:`frites.dataset.DatasetEphy`

dt = DatasetEphy(x, roi=roi, times=times)

###############################################################################
# Compute the pairwise connectivity
# ---------------------------------
#
# Once we have the dataset instance, we can then define an instance of workflow
# :class:`frites.workflow.WfConnComod`. This instance is then used to compute
# the pairwise connectivity

n_perm = 100  # number of permutations to compute
kernel = np.hanning(10)  # used for smoothing the MI

wf = WfConnComod(kernel=kernel)
mi, pv = wf.fit(dt, n_perm=n_perm, n_jobs=1)
print(mi)

###############################################################################
# Plot the result of the DataArray
# --------------------------------

# set to NaN everywhere it's not significant
is_signi = pv.data < .05
pv.data[~is_signi] = np.nan
pv.data[is_signi] = 1.02 * mi.data.max()

# plot each pair separately
plt.figure(figsize=(9, 7))
for n_r, r in enumerate(mi['roi'].data):
    # select the mi and p-values for the selected pair of roi
    mi_r, pv_r = mi.sel(roi=r), pv.sel(roi=r)
    plt.plot(times, mi_r, label=r, color=f"C{n_r}")
    if not np.isnan(pv_r.data).all():
        plt.plot(times, pv_r, color=f"C{n_r}", lw=4)
        x_txt = times[~np.isnan(pv_r)].mean()
        y_txt = 1.03 * mi.data.max()
        plt.text(x_txt, y_txt, r, color=f"C{n_r}", ha='center')
plt.legend()
plt.xlabel('Times')
plt.ylabel('Mi (bits)')
plt.title('Pairwise connectivity')
plt.show()
