"""
FIT: Feature specific information transfer
==========================================

This example illustrates how to compute Feature-specific Information
Transfer (FIT), quantifying how much information about a specific
feature flows between two regions. FIT merges the Wiener-Granger causality
principle with information-content specificity.
The theoretical background is described in [1] and the FIT is computed
using the :func:`frites.conn.conn_fit` function.
[1] Celotto M, Bím J, Tlaie A, De Feo V, Toso A, Lemke SM, Chicharro D,
Nili H, Bieler M, Hanganu-Opatz IL, Donner TH, Brovelli A, Panzeri S
(2023). An information-theoretic quantification of the content of
communication between brain regions. Advances in Neural Information
Processing Systems (NeurIPS), 36, 64213-64265
"""

import numpy as np
import xarray as xr

from frites.simulations import StimSpecAR
from frites.conn import conn_fit

from frites import set_mpl_style

import matplotlib.pyplot as plt
set_mpl_style()

###############################################################################
# Data simulation
# ---------------
#
# Here, we use an auto-regressive simulating a gamma increase.

net = False
avg_delay = False
ar_type = 'hga'
n_stim = 3
n_epochs = 400

ss = StimSpecAR()
ar = ss.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim, random_state=0)

print(ar)

plt.figure(figsize=(7, 8))
ss.plot(cmap='bwr')
plt.tight_layout()
plt.show()

###############################################################################
# Compute Feature specific information transfer
# -----------------------------------------
#
# Now we can use the simulated data to estimate the FIT.

# Compute the FIT

fit = conn_fit(ar, y='trials', roi='roi', times='times', mi_type='cd',
               max_delay=.3, net=net, verbose=False, avg_delay=avg_delay)

# Plot the results
fit.plot(x='times', col='roi')  # net = False

plt.show()
