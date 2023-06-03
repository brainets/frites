"""
Estimate interaction information
================================

This example illustrates how to compute the interaction information (II) to
investigate if pairs of brain regions are mainly carrying redundant or
synergistic information about task-related variables (e.g. stimulus type,
outcomes, learning rate etc.)
"""
import numpy as np
import xarray as xr

from frites.simulations import sim_single_suj_ephy
from frites.conn import conn_ii

from frites import set_mpl_style

import matplotlib.pyplot as plt
set_mpl_style()


###############################################################################
# Introduce redundant and synergistic informations
# ------------------------------------------------
#
# No we can inject in the simulated data redundancy and synergy. We first start
# by defining a task-related variable (e.g. the learning rate) and we will
# then inject this variable into the two first brain regions to simulate
# redundant coding. Then, to simulate synergy, we will inject half of the
# trials of the learning rate into one brain region and the other half of
# trials to a different brain region. That way, the two brain regions are going
# to be needed to fully decode the learning rate which is going to be reflected
# by synergistic interactions.

# simulate the brain data
n_roi = 4
n_epochs = 200
n_times = 1000
x = np.random.rand(n_epochs, n_roi, n_times)

# let's start by defining the task-related variable
y = np.random.rand(n_epochs)

# reshape it to inject it inside the different brain regions
y_repeated = np.tile(y.reshape(-1, 1), (1, 100))
y_repeated *= np.hanning(y_repeated.shape[-1]).reshape(1, -1)

# inject y inside the first and second brain regions to introduce redundancy
# between them
t_half = int(np.round(n_times / 2))
x[:, 0, t_half - 50:t_half + 50] += y_repeated
x[:, 1, t_half - 50:t_half + 50] += y_repeated

# now, inject half of the trials inside the third brain region and the second
# half of the trils in the fourth region. That way, we'll introduce synergy
# between them as the two brain regions are going o be needed to fully decode
# y
x[0:50:, 2, t_half - 50:t_half + 50] += y_repeated[0:50, ...]
x[50::, 3, t_half - 50:t_half + 50] += y_repeated[50::, ...]

# finally, merge everything inside a data array
times = (np.arange(n_times) - t_half + 50) / 512.
roi = ['roi_1', 'roi_2', 'roi_3', 'roi_4']
x = xr.DataArray(x, dims=('trials', 'roi', 'times'), coords=(y, roi, times))


###############################################################################
# Compute the interaction information
# -----------------------------------
#
# Now, we can estimate the interaction information. The II reflects whether
# pairs of brain regions are mainly carrying the same (i.e. redundant)
# information about the behavior or, if they are carrying complementary
# (i.e. synergy) information. Synergy is represented by a positive II and
# redundancy by a negative II.

# compute the II
ii = conn_ii(
    x, y, roi='roi', times='times', mi_type='cc'
)
print(ii)

# plot the result
fg = ii.plot(x='times', col='roi', col_wrap=3, size=4)
minmax = max(abs(ii.data.min()), abs(ii.data.max()))
plt.ylim(-minmax, minmax)
_ = [ax.axvline(0., color='C3') for ax in np.ravel(fg.axs)]
plt.gcf().suptitle(
    "Interaction information (II, in bits) of pairs of brain regions about y",
    fontsize=20, fontweight='bold', y=1.02)
plt.show()

# As we see in the results :
# - The II between roi_1 and roi_2 is redundant (II < 0)
# - The II between roi_3 and roi_4 is synergistic (II > 0)
# - The remaining pairs of brain regions are mainly carrying slightly redundant
#   information
