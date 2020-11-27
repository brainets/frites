"""
AR : simulate common driving input
==================================

This example illustrates an autoregressive model that simulates a common
driving input (i.e X->Y and X->Z) and how it is measured using the covariance
based Granger Causality
"""
import numpy as np

from frites.simulations import StimSpecAR
from frites.conn import conn_covgc

import matplotlib.pyplot as plt


###############################################################################
# Simulate 3 nodes 40hz oscillations
# ----------------------------------
#
# Here, we use the class :class:`frites.simulations.StimSpecAR` to simulate an
# stimulus-specific autoregressive model made of three nodes (X, Y and Z). This
# network simulates a transfer X->Y and X->Z. X is then called a common driving
# input for Y and Z


ar_type = 'osc_40_3'  # 40hz oscillations
n_stim = 3          # number of stimulus
n_epochs = 50       # number of epochs per stimulus

ss = StimSpecAR()
ar = ss.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim)
print(ar)

###############################################################################
# plot the network

plt.figure(figsize=(5, 4))
ss.plot_model()
plt.show()

###############################################################################
# plot the data

plt.figure(figsize=(7, 8))
ss.plot(cmap='bwr')
plt.tight_layout()
plt.show()

###############################################################################
# plot the power spectrum density (PSD)

plt.figure(figsize=(7, 8))
ss.plot(cmap='Reds', psd=True)
plt.tight_layout()
plt.show()


###############################################################################
# Compute the Granger-Causality
# -----------------------------
#
# We then compute and plot the Granger Causality. From the plot you can see
# that there's indeed an information transfer from X->Y and X->Z and, in
# addition, an instantaneous connectivity between Y.Z

dt = 50
lag = 5
step = 2
t0 = np.arange(lag, ar.shape[-1] - dt, step)
gc = conn_covgc(ar, roi='roi', times='times', dt=dt, lag=lag, t0=t0,
                n_jobs=-1)

# sphinx_gallery_thumbnail_number = 4
plt.figure(figsize=(12, 10))
ss.plot_covgc(gc)
plt.tight_layout()
plt.show()
