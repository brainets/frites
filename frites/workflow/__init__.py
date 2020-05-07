"""Workflow functions for assessing mutual information and statistics.

The following functions are pipelines designed for computing the mutual
information and statistics (fixed effect or random effect).
"""
from .wf_mi import WfMi  # noqa
from .wf_stats_ephy import WfStatsEphy  # noqa
from .wf_fit import WfFit  # noqa
from .wf_conn import WfConn  # noqa
