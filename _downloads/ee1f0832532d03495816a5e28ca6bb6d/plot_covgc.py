"""
Estimate the covariance-based Granger Causality
===============================================

This example illustrates how to compute single-trial covariance-based Granger
Causality.
"""
import numpy as np
from itertools import product

from frites.simulations import sim_single_suj_ephy
from frites.core import covgc

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


###############################################################################
# Simulate electrophysiological data
# ----------------------------------
#
# Let's start by simulating MEG / EEG electrophysiological data coming from
# a single subject. The output data of this single subject has a shape of
# (n_epochs, n_roi, n_times)

modality = 'meeg'
n_roi = 3
n_epochs = 10
n_times = 1000
x, roi, _ = sim_single_suj_ephy(n_epochs=n_epochs, n_times=n_times,
                                modality=modality, n_roi=n_roi, random_state=0)
times = np.linspace(-1, 1, n_times)

###############################################################################
# Simulate spatial correlations
# -----------------------------
#
# Bellow, we are simulating some distant correlations by injecting the
# activity of an ROI to another

# instantaneous between 0 and 2 (0.2)
x[:, [2], slice(200, 400)] += x[:, [0], slice(200, 400)]
# directed flow from 2 to 1 (2->1)
x[:, [1], slice(400, 600)] += x[:, [2], slice(400, 600)]
x[:, [2], slice(399, 599)] += x[:, [2], slice(400, 600)]
# directed flow from 0 to 1 (0->1)
x[:, [0], slice(600, 800)] += x[:, [1], slice(600, 800)]
x[:, [0], slice(599, 799)] += x[:, [0], slice(600, 800)]


###############################################################################
# Compute the covgc
# -----------------
#
# The covgc is going to be computed per trials, bewteen pairs of ROI and inside
# each of the temporal window

t0 = np.arange(100, 900, 10)
lag = 10
dt = 100
gc, pairs, roi_p, times_p = covgc(x, dt, lag, t0, times=times, roi=roi,
                                  n_jobs=1, output_type='dataarray')
# take the mean across trials
gc = gc.mean('trials')

plt.figure(figsize=(10, 8))
plt.subplot(311)
for r in roi_p:
    plt.plot(gc.times.data, gc.sel(roi=r, direction='x->y').T,
             label=r.replace('-', ' -> '))
plt.legend()
plt.subplot(312)
for r in roi_p:
    plt.plot(gc.times.data, gc.sel(roi=r, direction='y->x').T,
             label=r.replace('-', ' <- '))
plt.legend()
plt.subplot(313)
for r in roi_p:
    plt.plot(gc.times.data, gc.sel(roi=r, direction='x.y').T,
             label=r.replace('-', ' . '))
plt.legend()
plt.xlabel('Time')
plt.show()
