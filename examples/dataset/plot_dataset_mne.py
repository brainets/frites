"""
Define an electrophysiological dataset using MNE-Python structures
==================================================================

This example illustrates how to define a dataset using MNE-Python Epochs.
"""
import numpy as np

from mne import EpochsArray, create_info
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

###############################################################################
# MNE conversion to Epoch
# -----------------------
#
# Here, we convert our illustrative data into EpochsArray. With real data, you
# are probbaly going to have mne.Epochs objects which also going to work just
# the same

x_mne = []
for k in range(n_subjects):
    # create some informations
    info = create_info(ch[k].tolist(), sf)
    # create the Epoch of this subject
    epoch = EpochsArray(x[k], info, tmin=times[0], verbose=False)
    # finally, replace it in the original list
    x_mne.append(epoch)
print(x_mne[0])

###############################################################################
# Build the dataset
# -----------------
#
# Finally, we pass the data to the :class:`frites.dataset.DatasetEphy` class
# in order to create the dataset

dt = DatasetEphy(x_mne)
print(dt)

print('Time vector : ', dt.times)
print('ROI DataFrame\n: ', dt.df_rs)
