"""
Xarray : Saving the results
===========================

This example illustrates how to export and load your results. In details, we
are going to show how to save and reload a single DataArray such as a Dataset.
"""
import numpy as np
import xarray as xr
import pandas as pd
from time import sleep

###############################################################################
# Simulate data
# -------------
#
# lets start by creating two random spatio-temporal arrays

n_times = 30
n_roi = 7
times_vec = np.linspace(-1, 1, n_times)
roi_vec = np.array([f"roi_{k}" for k in range(n_roi)])

# xarray.DataArray conversion
arr_1 = xr.DataArray(np.random.rand(n_times, n_roi), dims=('times', 'roi'),
                     coords=(times_vec, roi_vec))
arr_2 = xr.DataArray(np.random.rand(n_times, n_roi), dims=('times', 'roi'),
                     coords=(times_vec, roi_vec))

# just add a few attributes to each array
arr_1.attrs['desc'] = "This is my first array"
arr_1.attrs['sf'] = 1024.
arr_2.attrs['desc'] = "This is my second array"
arr_2.attrs['sf'] = 512.

# note that you can also concatenate DataArray
arr_cat = xr.concat([arr_1, arr_2], 'roi')

###############################################################################
# Export and load a single DataArray
# ----------------------------------
#
# now we're going to save a single array and then reload it

# export a single array
arr_1.to_netcdf("first_array.nc")

# delete it
del arr_1
sleep(3)

# reload it
arr_1 = xr.load_dataarray("first_array.nc")
print(arr_1)

###############################################################################
# Export and load multiple DataArrays
# -----------------------------------
#
# it's also possible to export and reload multiple DataArrays at once. To do
# it, you can use a Dataset which is a container of DataArrays

# create a dataset
dat = xr.Dataset({'first': arr_1, 'second': arr_2})

# you can also slice the dataset and also add attributes to it
dat.attrs['desc'] = 'This is my dataset'
dat.attrs['sf'] = 256.

# export your dataset
dat.to_netcdf('full_dataset.nc')

# delete it
del dat
sleep(3)

# reload it
dat = xr.load_dataset("full_dataset.nc")
print(dat)

# finally, accessing array of a dataset is similar of using dictionary
arr_1 = dat['first']
arr_2 = dat['second']
