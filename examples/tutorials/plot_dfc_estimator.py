"""
Estimate the Dynamic Functional Connectivity
============================================

This tutorial illustrates how to compute the Dynamic Functional Connectivity
(DFC). In particular, we will adress :

* How to estimate the DFC at the single trial-level inside a unique time window
* How to estimate the DFC on sliding windows
* How to use different estimators (mutual-information, correlation and distance
  correlation)
* What are the strengths and weaknesses of each estimator
"""
import numpy as np
import xarray as xr

from frites.estimator import (GCMIEstimator, CorrEstimator, DcorrEstimator)
from frites.conn import conn_dfc, define_windows
from frites import set_mpl_style

import matplotlib.pyplot as plt
set_mpl_style()

# for reproducibility
np.random.seed(0)

###############################################################################
# Data simulation
# ---------------
#
# In this first section, we generate simulated data. We first use random data
# coming from several brain regions and then we introduce some correlations
# between the first two brain regions.

n_trials = 100
n_roi = 3
n_times = 250
trials = np.arange(n_trials)
roi = [f"r{n_r}" for n_r in range(n_roi)]
times = np.arange(n_times) / 64.
x = np.random.uniform(-1, 1, (n_trials, n_roi, n_times))

# positive correlation between samples [40, 60]
x[:, 0, 40:60] += .4 * x[:, 1, 40:60]
# negative correlation between samples [90, 110]
x[:, 0, 90:110] -= .4 * x[:, 1, 90:110]
# non-linear but monotone relationship between samples [140, 160]
x[:, 0, 140:160] += .4 * x[:, 1, 140:160] ** 3
# non-linear and non-monotone relationship between samples [190, 210]
x[:, 0, 190:210] += x[:, 1, 190:210] ** 2

###############################################################################
# .. note::
#     To summarize :
#
#         1. Electrophysiological data is a 3D array of shape
#            (100 trials, 3 brain regions, 250 time points)
#         2. Brain regions 0 and 1 are positively correlated between samples
#            [40, 60]
#         3. Brain regions 0 and 1 are negatively correlated between samples
#            [90, 110]
#         3. Brain regions 0 and 1 are positively correlated between samples
#            [140, 160] with a monotone but non-linear relationship
#         4. Brain regions 0 and 1 are positively correlated between samples
#            [190, 210] with a non-monotone and non-linear relationship

# dataarray transformation
x = xr.DataArray(x, dims=('trials', 'space', 'times'),
                 coords=(trials, roi, times))
print(x)

###############################################################################
# Computes the DFC in a single temporal window
# --------------------------------------------
#
# In the section we compute the DFC inside a single time-window. Actually, the
# DFC is going to be computed across the entire time-series

# compute the dfc
dfc = conn_dfc(x, times='times', roi='space')

print(dfc)

###############################################################################
# Computes the DFC on sliding windows
# -----------------------------------
#
# In this section, we are going to define sliding windows and then compute the
# DFC inside each one of them.

slwin_len = .5    # windows of length 500ms
slwin_step = .02  # 20ms step between each window (or 480ms overlap)

# define the sliding windows
sl_win = define_windows(times, slwin_len=slwin_len, slwin_step=slwin_step)[0]
print(sl_win)

# compute the DFC on sliding windows
dfc = conn_dfc(x, times='times', roi='space', win_sample=sl_win)

# takes the mean over trials
dfc_m = dfc.mean('trials').squeeze()

# plot the mean over trials
dfc_m.plot.line(x='times', hue='roi')
plt.title(dfc.name), plt.ylabel('DFC')
plt.show()

###############################################################################
# Comparison of several estimators
# --------------------------------
#
# By default, the `conn_dfc` function uses the Gaussian-Copula
# Mutual-Information (GCMI) estimator. However, the `conn_dfc` function allows
# to provide other estimators as soon as it is made for computing information
# between two continuous variables (`mi_type='cc'`). In this final section, we
# are going to use different estimators, especially the standard correlation
# and the distance correlation.

est_mi = GCMIEstimator('cc', copnorm=None, biascorrect=False)
est_corr = CorrEstimator()
est_dcorr = DcorrEstimator()

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 12))

for n_e, est in enumerate([est_mi, est_corr, est_dcorr]):
    # compute the dfc
    dfc = conn_dfc(x, times='times', roi='space', win_sample=sl_win,
                   estimator=est)

    # take the mean across trials
    dfc_m = dfc.mean('trials')

    # plot the result
    plt.sca(axs[n_e])
    dfc_m.plot.line(x='times', hue='roi', ax=plt.gca())
    plt.title(dfc.name)
    plt.ylabel('DFC')
    if n_e != 2: plt.xlabel('')

plt.tight_layout()
plt.show()

###############################################################################
# .. note::
#     To summarize :
#
#         1. GCMI estimators offers a great sensibility. However, the
#            mutual-information is unsigned and therefore, negative
#            correlations are captured as positive information (cf. second
#            bump). In addition, non-monotone relations are not well captured
#         2. On the other hand, the correlation clearly extract both positive
#            and negative correlations however, non-monotone relationships are
#            missed. Finally, the correlation is probably not as sensible as
#            the GCMI
#         3. Finally, the distance correlation captures all relations but as
#            the GCMI, it is an unsigned measure, missing negative
#            correlations. It is the most powerful estimator however, it is
#            also slower to compute.
