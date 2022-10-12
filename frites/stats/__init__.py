"""Statistical methods.

This submodule contains a collection of statistical internal methods divided
into two categories :

1. **Random efect estimation :** t-test related functions
2. **P-values correction for multiple comparisons :** test-wise and cluster
   based corrections

Most of those stastical functions are using
`MNE Python <https://mne.tools/stable/index.html>`_
"""
from .stats_mcp import (testwise_correction_mcp, cluster_correction_mcp,  # noqa
                        cluster_threshold)
from .stats_nonparam import (permute_mi_vector, permute_mi_trials,  # noqa
                             bootstrap_partitions, dist_to_ci,
                             confidence_interval, trial_swap_surrogates)
from .stats_param import (ttest_1samp, rfx_ttest)  # noqa
