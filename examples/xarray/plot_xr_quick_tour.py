"""
Xarray : Quick tour
===================

Basically, this example gives a very small introduction to Xarray (very small).
We illustrate how to define a DataArray container, access its components,
perform some of the basic operations and slicing / indexing.
"""
import numpy as np
import xarray as xr
import pandas as pd

###############################################################################
# Simulate data
# -------------
#
# lets start by creating a random spatio-temporal array

n_times = 30
n_roi = 7
times_vec = np.linspace(-1, 1, n_times)
roi_vec = np.array([f"roi_{k}" for k in range(n_roi)])
np_data = np.random.rand(n_times, n_roi)
print(np_data.shape)
print('*' * 79)

###############################################################################
# Xarray conversion and access to the internal components
# -------------------------------------------------------
#
# A DataArray is a container (like a well known numpy array) except that you
# can add a label to each coordinate. To this end, the input `dims` is a tuple
# that describes the dimension names and `coords` describes the value along
# this coordinate

# let's convert it to a DataArray
da_data = xr.DataArray(np_data, dims=('times', 'roi'),
                       coords=(times_vec, roi_vec))
print(da_data.shape)
print(da_data)
print('*' * 79)

# if you want to get the dimension names and values
print(f'Dimension names : {da_data.dims}')
print(f'Dimension values : {da_data.coords}')
print(f"Data of a specific dimension : {da_data.roi.data}")
print('*' * 79)

# if you want to get the original NumPy array enter the following :
da_data.data

# if you want to change the values of a coordinate
da_data['roi'] = np.array([f"roi_{k % 3}" for k in range(n_roi)])
print(f"New ROI names : {da_data.roi.data}")
print('*' * 79)

# if you need to compute or get the min / max / mean across a specific
# dimension
da_data.min('times')  # minimum across time points
da_data.max('times')  # maximum across time points
da_data.mean('roi')   # mean across all ROI

# similarly to Pandas, it's also possible to group along a dimension and then
# take the mean. For example, here's how to group and mean by roi names
da_m = da_data.groupby('roi').mean('roi')
print(da_m)
print('*' * 79)


###############################################################################
# Xarray slicing and indexing
# ---------------------------
#
# Now we show how to slice the container

# select a single specific ROI based on it's name
da_data.sel(roi='roi_0')

# select a time range
da_time_slice = da_data.sel(times=slice(-.5, .5))
print(f"Temporal selection : {da_time_slice.coords}")
print('*' * 79)

# off course, spatio-temporal selection is also supported
da_st = da_data.sel(times=slice(-.5, .5), roi='roi_1')
print(f"Spatio-temporal selection : {da_st.coords}")
print('*' * 79)

# you can also slice according to indices
da_isel = da_data.isel(times=slice(10, 20))
print(f"Integer selection : {da_isel.coords}")
print('*' * 79)

# however, if you want for example select multiple items based on their names,
# you have to use booleans. Here's a small example that's using Pandas
roi = da_data.roi.data
use_roi = ['roi_0', 'roi_2']
is_roi = pd.Series(roi).str.contains('|'.join(use_roi))
da_mi = da_data.isel(roi=is_roi)
print(f"Multi-items selection : {da_mi.coords}")

###############################################################################
# Xarray attributes
# -----------------
#
# One of the nice features of DataArray is that it supporting setting
# attributes. Therefore you can add, for example, the parameters that describe
# your analysis

# adding a few string attributes
da_data.attrs['inference'] = 'ffx'
da_data.attrs['stats'] = 'cluster-based'
da_data.attrs['description'] = """Here's a small description of the analysis
I'm currently running. Trying to find a difference between condition 1. vs 2.
"""

# you can also add vectors (but not arrays) to the attributes
da_data.attrs['vector'] = np.arange(30)

# however, "None" seems to pose a problem when saving the results. Therefore,
# one quick way to solve this is simply to convert it into a string
da_data.attrs['none_problem'] = str(None)

print(da_data)

###############################################################################
# Xarray to an other format
# -------------------------
#
# Finally, we quickly illustrate how to convert a DataArray into, for example,
# a pandas.DataFrame

print(da_data.to_pandas())