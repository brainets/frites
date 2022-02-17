"""
Mutual-information at the contact level
=======================================

This example illustrates how to compute the mutual information inside each
brain region and also by taking into consideration the information at a lower
anatomical level (e.g sEEG contact, MEG sensor etc.).
"""
import numpy as np
import pandas as pd
import xarray as xr

from frites.dataset import DatasetEphy
from frites.workflow import WfMi
from frites.core import mi_nd_gg
from frites import set_mpl_style

import matplotlib.pyplot as plt
set_mpl_style()

###############################################################################
# I(Continuous; Continuous) case
# ------------------------------
#
# Let's start by simulating by using random data in combination with normal
# distributions. To explain why it could be interesting to consider the
# information at the single contact level, we generate the data coming from one
# single brain region ('roi_0') but with two contacts inside ('c1', 'c2').
# For the first the contact, the brain data are going to be positively
# correlated with the normal distribution while the second contact is going to
# have negative correlations. If you concatenate the data of the two contacts,
# the mix of positive and negative correlations break the monotonic
# assumption of the GCMI. In that case, it's better to compute the MI per
# contact

n_suj = 3
n_trials = 20
n_times = 100
half = int(n_trials / 2)
times = np.arange(n_times)

x, y, roi = [], [], []
for suj in range(n_suj):
    # initialize subject's data with random noise
    _x = np.random.rand(n_trials, 2, n_times)
    # normal continuous regressor
    _y = np.random.normal(size=(n_trials,))

    # first contact has positive correlations
    _x[:, 0, slice(30, 70)] += _y.reshape(-1, 1)
    # second contact has negative correlations
    _x[:, 1, slice(30, 70)] -= _y.reshape(-1, 1)

    x += [_x]
    y += [_y]
    roi += [np.array(['roi_0', 'roi_0'])]

# now, compute the mi with default parameters
ds = DatasetEphy(x, y=y, roi=roi, times=times, agg_ch=True)
mi = WfMi(mi_type='cc').fit(ds, mcp='noperm')[0]

# compute the mi at the contact level
ds = DatasetEphy(x, y=y, roi=roi, times=times, agg_ch=False)
mi_c = WfMi(mi_type='ccd').fit(ds, mcp='noperm')[0]

# plot the comparison
plt.figure()
plt.plot(times, mi, label="MI across contacts")
plt.plot(times, mi_c, label="MI at the contact level")
plt.legend()
plt.title('I(C; C)')
plt.show()


###############################################################################
# I(Continuous; Discret) case
# ---------------------------
#
# Same example as above except that this time the MI is compute between the
# data and a discret variable

x, y, roi = [], [], []
for suj in range(n_suj):
    # initialize subject's data with random noise
    _x = np.random.rand(n_trials, 2, n_times)
    # define a positive and negative offsets of 1
    _y_pos, _y_neg = np.full((half, 1), 1.), np.full((half, 1), -1.)

    # first contact / first half trials : positive offset
    _x[0:half, 0, slice(30, 70)] += _y_pos
    # first contact / second half trials : negative offset
    _x[half::, 0, slice(30, 70)] += _y_neg
    # second contact / first half trials : negative offset
    _x[0:half, 1, slice(30, 70)] += _y_neg
    # second contact / second half trials : positive offset
    _x[half::, 1, slice(30, 70)] += _y_pos

    x += [_x]
    y += [np.array([0] * half + [1] * half)]
    roi += [np.array(['roi_0', 'roi_0'])]
times = np.arange(n_times)

# now, compute the mi with default parameters
ds = DatasetEphy(x, y=y, roi=roi, times=times)
mi = WfMi(mi_type='cd').fit(ds, mcp='noperm')[0]

# compute the mi at the contact level
ds = DatasetEphy(x, y=y, roi=roi, times=times, agg_ch=False)
mi_c = WfMi(mi_type='cd').fit(ds, mcp='noperm')[0]

# plot the comparison
plt.figure()
plt.plot(times, mi, label="MI across contacts")
plt.plot(times, mi_c, label="MI at the contact level")
plt.legend()
plt.title('I(C; D)')
plt.show()


###############################################################################
# I(Continuous ; Continuous | Discret) case
# -----------------------------------------
#
# Same example as above except that this time the MI is compute between the
# data and a discret variable


x, y, z, roi = [], [], [], []
for suj in range(n_suj):
    # initialize subject's data with random noise
    _x = np.random.rand(n_trials, 2, n_times)
    # define a positive and negative correlations
    _y_pos = np.random.normal(loc=1, size=(half))
    _y_neg = np.random.normal(loc=-1, size=(half))
    _y = np.r_[_y_pos, _y_neg]
    _z = np.array([0] * half + [1] * half)

    # first contact / first half trials : positive offset
    _x[0:half, 0, slice(30, 70)] += _y_pos.reshape(-1, 1)
    # first contact / second half trials : negative offset
    _x[half::, 0, slice(30, 70)] += _y_neg.reshape(-1, 1)
    # second contact / first half trials : negative offset
    _x[0:half, 1, slice(30, 70)] += _y_neg.reshape(-1, 1)
    # second contact / second half trials : positive offset
    _x[half::, 1, slice(30, 70)] += _y_pos.reshape(-1, 1)

    x += [_x]
    y += [_y]
    z += [_z]
    roi += [np.array(['roi_0', 'roi_0'])]
times = np.arange(n_times)

# now, compute the mi with default parameters
ds = DatasetEphy(x, y=y, z=z, roi=roi, times=times)
mi = WfMi(mi_type='ccd').fit(ds, mcp='noperm')[0]

# compute the mi at the contact level
ds = DatasetEphy(x, y=y, z=z, roi=roi, times=times, agg_ch=False)
mi_c = WfMi(mi_type='ccd').fit(ds, mcp='noperm')[0]

# plot the comparison
plt.figure()
plt.plot(times, mi, label="MI across contacts")
plt.plot(times, mi_c, label="MI at the contact level")
plt.legend()
plt.title('I(C; C | D)')
plt.show()

###############################################################################
# Xarray definition with multi-indexing
# -------------------------------------
#
# Finally, we show below how to use Xarray in combination with pandas
# multi-index to define an electrophysiological dataset

x = []
for suj in range(n_suj):
    # initialize subject's data with random noise
    _x = np.random.rand(n_trials, 2, n_times)
    # normal continuous regressor
    _y = np.random.normal(size=(n_trials,))

    # first contact has positive correlations
    _x[:, 0, slice(30, 70)] += _y.reshape(-1, 1)
    # second contact has negative correlations
    _x[:, 1, slice(30, 70)] -= _y.reshape(-1, 1)
    # roi and contacts definitions
    _roi = np.array(['roi_0', 'roi_0'])
    _contacts = np.array(['c1', 'c2'])

    # multi-index definition
    midx = pd.MultiIndex.from_arrays([_roi, _contacts],
                                     names=('parcel', 'contacts'))
    # xarray definition
    _x_xr = xr.DataArray(_x, dims=('trials', 'space', 'times'),
                         coords=(_y, midx, times))

    x += [_x_xr]

# finally, define the electrophysiological dataset and specify the dimension
# names to use
ds_xr = DatasetEphy(x, y='trials', roi='parcel', agg_ch=False, times='times')
print(ds_xr)
