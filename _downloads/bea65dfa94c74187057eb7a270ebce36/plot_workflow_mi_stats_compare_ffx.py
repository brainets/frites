"""
Compare within-subjects statistics when computing mutual information
====================================================================

This example illustrates how to define and run a workflow for computing
mutual information and evaluate stastitics. Inference are made within-subjects
(fixed effect = ffx). 
"""
import numpy as np

from frites.simulations import sim_multi_suj_ephy, sim_mi_cc
from frites.dataset import DatasetEphy
from frites.workflow import WorkflowMiStats

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


###############################################################################
# Simulate electrophysiological data
# ----------------------------------
#
# Let's start by simulating MEG / EEG electrophysiological data coming from
# multiple subjects using the function
# :func:`frites.simulations.sim_multi_suj_ephy`. As a result, the `data` output
# is a list of length `n_subjects` of arrays, each one with a shape of
# n_epochs, n_sites, n_times

modality = 'meeg'
n_subjects = 5
n_epochs = 100
n_times = 100
data, roi, time = sim_multi_suj_ephy(n_subjects=n_subjects, n_epochs=n_epochs,
                                     n_times=n_times, modality=modality,
                                     random_state=0)

###############################################################################
# Simulate mutual information
# ---------------------------
#
# Once the data have been created, we simulate an increase of mutual
# information by creating a continuous variable `y` using the function
# :func:`frites.simulations.sim_mi_cc`. This allows to simulate model-based
# analysis by computing $I(data; y)$ where `data` and `y` are two continuous
# variables

y, _ = sim_mi_cc(data, snr=.1)

###############################################################################
# Create an electrophysiological dataset
# --------------------------------------
#
# Now, we use the :class:`frites.dataset.DatasetEphy` in order to create a
# compatible electrophysiological dataset

dt = DatasetEphy(data, y, roi=roi, times=time, verbose=False)

###############################################################################
# Define the workflow
# -------------------
#
# We now define the workflow for computing mi and evaluate statistics using the
# class :class:`frites.workflow.WorkflowMiStats`. Here, the type of mutual
# information to perform is 'cc' between it's computed between two continuous
# variables. And we also specify the inference type 'ffx' for fixed-effect

mi_type = 'cc'
inference = 'ffx'
wf = WorkflowMiStats(mi_type, inference)

###############################################################################
# Compute the mutual information and statistics
# ---------------------------------------------

# list of fixed-effect methods to test
ffx_methods = ('ffx_maxstat', 'ffx_fdr', 'ffx_bonferroni',
               'ffx_cluster_maxstat', 'ffx_cluster_fdr',
               'ffx_cluster_bonferroni', 'ffx_cluster_tfce')

n_perm = 1000
n_jobs = 1
pvalues = []
plt.figure(figsize=(10, 8))
for ffx in ffx_methods:
    mi, pvalues = wf.fit(dt, n_jobs=n_jobs, n_perm=n_perm, stat_method=ffx,
                         output_type='array')
    # set to 1. everywhere p-values are not significants
    pvalues[pvalues >= .05] = 1.

    plt.subplot(212)
    plt.plot(time, pvalues.squeeze(), label=ffx)
    plt.xlabel('Time (s)'), plt.ylabel("P-values")
    plt.title("P-values (ffx)")
plt.legend()

plt.subplot(211)
plt.plot(time, mi.squeeze())
plt.xlabel('Time (s)'), plt.ylabel("Bits")
plt.title("Mutual information I(C; C)")
plt.autoscale(tight=True)

plt.show()