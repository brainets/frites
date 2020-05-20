"""
Build an electrophysiological dataset
=====================================

In Frites, a dataset is a structure for grouping the electrophysiological data
(e.g MEG / EEG / Intracranial) coming from multiple subjects. In addition,
some basic operations can also be performed (like slicing, smoothing etc.). In
this example we illutrate how to define a dataset using NumPy arrays.
"""
import numpy as np

from frites.dataset import DatasetEphy

import matplotlib.pyplot as plt

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
n_times = 1000
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

###############################################################################
# Create the dataset
# ------------------
#
# The creation of the dataset is performed using the class
# :class:`frites.dataset.DatasetEphy`

dt = DatasetEphy(x.copy(), roi=ch, times=times)
print(dt)

plt.plot(dt.times, dt.x[0][:, 0, :].T)
plt.xlabel('Times')
plt.title('Electrophysiological data of the first subject, for the first '
          'channel')
plt.show()

###############################################################################
# Data smoothing
# --------------
#
# If you have MNE-Python installed, you can also smooth the data using
# :class:`frites.dataset.DatasetEphy.savgol_filter`. One important thing is
# that operations are performed inplace, which means that once launched, the
# data are modified inside the dataset without copy

# high cut-off frequency at 4Hz
dt.savgol_filter(4)

plt.plot(dt.times, dt.x[0][:, 0, :].T)
plt.xlabel('Times')
plt.title('Smoothed dataset')
plt.show()

###############################################################################
# Data resampling
#
# Still using MNE-Python, you can also resample the dataset using 
# :class:`frites.dataset.DatasetEphy.resample`

# resample the dataset using a new sampling rate of 256Hz
dt.resample(256)

plt.plot(dt.times, dt.x[0][:, 0, :].T)
plt.xlabel('Times')
plt.title('Resampled dataset')
plt.show()

###############################################################################
# Spatio-temporal slicing
# -----------------------
#
# The dataset also supports some basic slicing operations through time and
# space. Slicing is still performed inplace

# temporal selection between [0.25, 1.75]
dt[0.25:1.75, :]  # the ':' symbol means that we are selecting every channel

plt.plot(dt.times, dt.x[0][:, 0, :].T)
plt.xlabel('Times')
plt.title('Temporal slicing')
plt.show()

# spatial selection of channels ch_0 and ch_1
dt[:, ['ch_0', 'ch_1']]
print(dt.roi)
