"""
Estimate comodulations between brain areas
==========================================

This example illustrates how to estimate the instantaneous comodulations using
mutual information between pairwise ROI and also perform statistics.
"""
import numpy as np
from itertools import product

from frites.simulations import sim_multi_suj_ephy
from frites.dataset import DatasetEphy
from frites.workflow import WfComod

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


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
# :class:`frites.workflow.WfComod`. This instance is then used to compute the
# pairwise connectivity

n_perm = 100  # number of permutations to compute
kernel = np.hanning(10)  # used for smoothing the MI

wf = WfComod(kernel=kernel)
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
for s, t in product(mi.source.data, mi.target.data):
    if s == t: continue
    # select the mi and p-values for the (source, target)
    mi_st = mi.sel(source=s, target=t)
    pv_st = pv.sel(source=s, target=t)
    color = np.random.rand(3,)
    plt.plot(times, mi_st, label=f"{s}-{t}", color=color)
    plt.plot(times, pv_st, color=color, lw=4)
    if not np.isnan(pv_st.data).all():
        x_txt = times[~np.isnan(pv_st)].mean()
        y_txt = 1.03 * mi.data.max()
        plt.text(x_txt, y_txt, f"{s}-{t}", color=color, ha='center')
plt.legend()
plt.xlabel('Times')
plt.ylabel('Mi (bits)')
plt.title('Pairwise connectivity')
plt.show()
