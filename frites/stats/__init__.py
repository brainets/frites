"""Fixed and random effect methods."""
from .stats_ffx import (ffx_maxstat, ffx_fdr, ffx_bonferroni,  # noqa
                        ffx_cluster_maxstat, ffx_cluster_fdr,
                        ffx_cluster_bonferroni, ffx_cluster_tfce)
from .stats_rfx import (rfx_cluster_ttest, rfx_cluster_ttest_tfce)  # noqa
from .stats_cluster import temporal_clusters_permutation_test  # noqa

STAT_FUN = dict(  # noqa
    ffx=dict(
    ffx_maxstat=ffx_maxstat,
    ffx_fdr=ffx_fdr,
    ffx_bonferroni=ffx_bonferroni,
    ffx_cluster_maxstat=ffx_cluster_maxstat,
    ffx_cluster_fdr=ffx_cluster_fdr,
    ffx_cluster_bonferroni=ffx_cluster_bonferroni,
    ffx_cluster_tfce=ffx_cluster_tfce
    ),
    rfx=dict(
    rfx_cluster_ttest=rfx_cluster_ttest,
    rfx_cluster_ttest_tfce=rfx_cluster_ttest_tfce
    )
)
