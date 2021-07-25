"""Workflows performing a series of analysis.

This submodule contains several pipelines of analysis. It is divided into two
categories of workflows :

1. **Task-related workflows :** two-steps pipelines where the first step
   consist in quantifying an effect size (either on local brain activity either
   on connectivity) and the second step consisting in computing group-level
   statistics
2. **Statistics workflows :** pure statistics workflows dedicated to
   information-based measures
"""
# task related workflows
from .wf_mi import WfMi  # noqa
from .wf_mi_combine import WfMiCombine  # noqa
# connectivity workflows
from .wf_conn_comod import WfConnComod  # noqa
# statistical workflows
from .wf_stats import WfStats  # noqa
