"""
Define an electrophysiological dataset using Xarray
===================================================

This example illustrates how to define a dataset using Xarray. If you don't
know this library, we can simplify by saying that it provides containers that
accept arrays but you can also labelize your dimensions. Another way of seeing
it, pandas is mostly made for tables (i.e 2D arrays) while Xarray provide
almost the same functionalities but for multi-dimensional arrays.

"""
import numpy as np
import pandas as pd

from xarray import DataArray
from frites.dataset import DatasetEphy
from frites import set_mpl_style

import matplotlib.pyplot as plt
set_mpl_style()

###############################################################################
# Create artificial data
# ----------------------
#
# We start by creating some random data for several subjects. To do that, each
# subject is going have a 3 dimensional array of shape
# (n_epochs, n_channels, n_times). Then, all of the arrays are grouped together
# in a list of length (n_subjects,)

n_subjects = 5
n_epochs = 10
n_channels = 5
n_times = 100
sf = 512

x, ch = [], []
for k in range(n_subjects):
    # generate single subject data
    x_suj = np.random.rand(n_epochs, n_channels, n_times)
    # generate some random channel names
    ch_suj = np.array([f"ch_{r}" for r in range(n_channels)])
    # concatenate in a list
    x.append(x_suj)
    ch.append(ch_suj)
# finally lets create a time vector
times = np.arange(n_times) / sf
epochs = np.arange(n_epochs)

###############################################################################
# Xarray conversion to DataArray
# ------------------------------
#
# Here, we convert the NumPy arrays to xarray.DataArray

x_xr = []
for k in range(n_subjects):
    # DataArray conversion
    arr_xr = DataArray(x[k], dims=('epochs', 'channels', 'times'),
                       coords=(epochs, ch[k], times))
    # finally, replace it in the original list
    x_xr.append(arr_xr)
print(x_xr[0])

###############################################################################
# Build the dataset
# -----------------
#
# Finally, we pass the data to the :class:`frites.dataset.DatasetEphy` class
# in order to create the dataset

# here, we specify to the DatasetEphy class that the roi dimension is actually
# called 'channels' in the DataArray and the times dimension is called 'times'
dt = DatasetEphy(x_xr, roi='channels', times='times')
print(dt)

print('Time vector : ', dt.times)
print('ROI\n: ', dt.df_rs)

###############################################################################
# MultiIndex support
# ------------------
#
# DataArray also supports multi-indexing of a single dimension.

# create a continuous regressor (prediction error, delta P etc.)
dp = np.random.uniform(-1, 1, (n_epochs,))
# create a discret variable (e.g experimental conditions)
cond = np.array([0] * 5 + [1] * 5)

# now, create a multi-index using pandas
midx = pd.MultiIndex.from_arrays((dp, cond), names=('dp', 'blocks'))

# convert again the original arrays but this time, the epoch dimension is going
# to be a multi-index
x_xr = []
for k in range(n_subjects):
    # DataArray conversion
    arr_xr = DataArray(x[k], dims=('epochs', 'channels', 'times'),
                       coords=(midx, ch[k], times))
    # finally, replace it in the original list
    x_xr.append(arr_xr)
print(x_xr[0])

# finally, when you create your dataset you can also specify the y and z inputs
# by providing their names in the DataArray
dt = DatasetEphy(x_xr, roi='channels', times='times', y='dp', z='blocks')
print(dt)
