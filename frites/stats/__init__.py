"""Fixed and random effect methods."""
from .stats_ffx import (ffx_maxstat, ffx_cluster_fdr, ffx_cluster_bonferroni,  # noqa
                        ffx_cluster_tfce)

STAT_FUN = dict(  # noqa
    ffx_maxstat=ffx_maxstat,
    ffx_cluster_fdr=ffx_cluster_fdr,
    ffx_cluster_bonferroni=ffx_cluster_bonferroni,
    ffx_cluster_tfce=ffx_cluster_tfce
)
