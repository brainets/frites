"""
Information transfer about a continuous variable
================================================

This example illustrates how to compute the amount of information that is sent
from one region to another about a specific continuous feature. For further
details, see Bim et al. 2019 :cite:`bim2019non`
"""
import numpy as np

from frites.simulations import sim_distant_cc_ms
from frites.dataset import DatasetEphy
from frites.workflow import WfFit

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


###############################################################################
# Simulate electrophysiological data
# ----------------------------------
#
# Let's start by simulating data coming from multiple subjects using the
# function :func:`frites.simulations.sim_distant_cc_ms`. As a result, the `x`
# output is a list of length `n_subjects` of arrays, each one with a shape of
# n_epochs, n_sites, n_times

n_subjects = 5
n_epochs = 200
x, y, roi, times = sim_distant_cc_ms(n_subjects, n_epochs=n_epochs)

###############################################################################
# Define the electrophysiological dataset
# ---------------------------------------
#
# Now we define an instance of :class:`frites.dataset.DatasetEphy`

ds = DatasetEphy(x, y, roi=roi, times=times)

###############################################################################
# Compute the bidirectionnal information transfer
# -----------------------------------------------
#
# Once we have the dataset instance, we can then define an instance of workflow
# :class:`frites.workflow.WfFit`. This instance is used to compute the
# information transfer

wf = WfFit()
mi, _ = wf.fit(ds, n_perm=10)
print(mi)

# when `net=False` it means that the information transfer is directed which
# means we can either look at the amount of informations sent from roi_0 to
# roi_1 or roi_1 to roi_0.
it_0_to_1 = mi.sel(source='roi_0', target='roi_1')
it_1_to_0 = mi.sel(source='roi_1', target='roi_0')

plt.plot(it_0_to_1, label='roi_0 -> roi_1')
plt.plot(it_1_to_0, label='roi_1 -> roi_0')
plt.title('Bidirectionnal FIT')
plt.xlabel('Time'), plt.ylabel('MI (bits)')
plt.show()

###############################################################################
# Compute the unidirectionnal information transfer
# ------------------------------------------------
#
# Note that you can also compute the unidirectionnal FIT which is define as the
# difference between `FIT(source - >target) - FIT(target -> source)`.

wf = WfFit()
mi, _ = wf.fit(ds, net=True, n_perm=10)
print(mi)
it_net = mi.sel(source='roi_0', target='roi_1')

plt.plot(it_net, label='roi_0 <-> roi_1')
plt.title('Unidirectionnal FIT')
plt.xlabel('Time'), plt.ylabel('MI (bits)')
plt.legend()
plt.show()