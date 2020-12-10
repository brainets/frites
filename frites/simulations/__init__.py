"""Simulate data for testing Frites functions.

This submodule contains a collection of functions to simulate data in order to
test Frites functions and workflows. There are several ways to simulate data :

1. **Using an auto-regressive model :** this method can be used to simulate
   a task-related brain network (i.e with local activity and information flow
   modulated by a stimulus)
2. **Using gaussian variables :** in that case, the generated data are feature
   specific but only at the node level
"""
# Documented functions
# --------------------

# task-related brain network
from .sim_ar import StimSpecAR  # noqa
# gaussian-based task-related functions
from .sim_local_mi import (sim_local_cc_ss, sim_local_cc_ms,  # noqa
                           sim_local_cd_ss, sim_local_cd_ms,
                           sim_local_ccd_ms, sim_local_ccd_ss)

# Undocumented functions
# ----------------------

"""
Functions without documentation but used internally
"""

from .sim_mi import (sim_mi_cc, sim_mi_cd, sim_mi_ccd)  # noqa
# simulate single and multi subjects electrophysiological data
from .sim_generate_data import (sim_single_suj_ephy, sim_multi_suj_ephy)  # noqa
