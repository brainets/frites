"""
Estimate dynamic functional connectivity
========================================

This example illustrates how to compute the dynamic functional connectivity
(DFC) using the mutual information (MI). This type of connectivity is computed
for each trial either inside a single window or across multiple windows.
"""
import numpy as np
from itertools import product

from frites.simulations import sim_single_suj_ephy
from frites.core import dfc_gc
from frites.utils import define_windows, plot_windows

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
n_epochs = 50
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

x[:, [1], slice(100, 400)] += x[:, [0], slice(100, 400)]
x[:, [2], slice(600, 800)] += x[:, [1], slice(600, 800)]
print(f'Corr 1 : {roi[0]}-{roi[1]} between [{times[100]}-{times[400]}]')
print(f'Corr 2 : {roi[2]}-{roi[1]} between [{times[600]}-{times[800]}]')

###############################################################################
# Define sliding windows
# ----------------------
#
# Next, we define, and plot sliding windows in order to compute the DFC on
# consecutive time windows. In this example we used windows of length 100ms
# and 5ms between each consecutive windows

slwin_len = .1    # 100ms window length
slwin_step = .08  # 80ms between consecutive windows
win_sample = define_windows(times, slwin_len=slwin_len,
                            slwin_step=slwin_step)[0]
times_p = times[win_sample].mean(1)

plt.figure(figsize=(10, 8))
plot_windows(times, win_sample, title='Sliding windows')
plt.ylim(-1, 1)
plt.show()

###############################################################################
# Compute the DFC
# ---------------
#
# The DFC is going to be computed per trials, bewteen pairs of ROI and inside
# each of the temporal window

# compute DFC
dfc, pairs, roi_p = dfc_gc(x, times, roi, win_sample)

# sphinx_gallery_thumbnail_number = 2
plt.figure(figsize=(10, 8))
plt.plot(times_p, dfc.mean(0).T)
plt.xlabel('Time')
plt.title("DFC between pairs of roi")
plt.show()

