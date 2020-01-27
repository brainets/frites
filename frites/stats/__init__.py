"""Fixed and random effect methods."""
from .stats_cluster import (temporal_clusters_permutation_test,  # noqa
                            cluster_threshold)
from .stats_param import (ttest_1samp, rfx_ttest)  # noqa
from .stats_mcp import (permutation_mcp_correction)  # noqa
