"""
PID: Decomposing the information carried by pairs of brain regions
==================================================================

This example illustrates how to decompose the information carried by pairs of
brain regions about a behavioral variable `y` (e.g. stimulus, outcome, learning
curve, etc.). Here, we use the Partial Information Decomposition (PID) that
leads four non-negative and exclusive atoms of information:
- The unique information carried by the first brain region about `y`
- The unique information carried by the second brain region about `y`
- The redundant information carried by both regions about `y`
- The synergistic r complementary information carried by both regions about `y`
"""
import numpy as np
import xarray as xr

from frites.simulations import StimSpecAR
from frites.conn import conn_pid

from frites import set_mpl_style

import matplotlib.pyplot as plt
set_mpl_style()


###############################################################################
# Data simulation
# ---------------
#
# Let's simulate some data. Here, we use an auto-regressive simulating a gamma
# increase. The gamma increase is modulated according to two conditions.

ar_type = 'hga'
n_stim = 2
n_epochs = 300

ss = StimSpecAR()
ar = ss.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim)

print(ar)

plt.figure(figsize=(7, 8))
ss.plot(cmap='bwr')
plt.tight_layout()
plt.show()


###############################################################################
# Compute Partial Information Decomposition
# -----------------------------------------
#
# Now we can use the simulated data to estimate the PID. Here, we'll try to
# decompose at each time point, the information carried by pairs of brain
# regions about the two conditions.

# compute the PID
infotot, unique, redundancy, synergy = conn_pid(
    ar, 'trials', roi='roi', times='times', mi_type='cd', verbose=False
)

# plot the results
infotot.plot(color='C3', label=r"$Info_{Tot}$", linestyle='--')
redundancy.plot(color='C0', label=r"$Redundancy_{XY}$")
synergy.plot(color='C1', label=r"$Synergy_{XY}$")
unique.sel(roi='x').squeeze().plot(color='C4', label=r"$Unique_{X}$")
unique.sel(roi='y').squeeze().plot(color='C5', label=r"$Unique_{Y}$")
plt.legend()
plt.ylabel("Information [Bits]")
plt.axvline(0., color='C3', lw=2)
plt.title("Decomposition of the information carried by a pair of brain regions"
          "\nabout a task-related variable", fontweight='bold')
plt.show()

###############################################################################
# from the plot above, we can see that:
# 1. The total information carried by the pairs of regions (Info_{Tot})
# 2. At the beginning, a large portion of the information is carried by the
#    first brain region (Unique_{X})
# 3. Then we can see a superimposition of redundancy (Redundancy_{XY}) and
#    synergy (Synergy_{XY}) carried by both regions
# 4. Finally, later in time most of the information is carried by the second
#    brain region Y (Unique_{X})
