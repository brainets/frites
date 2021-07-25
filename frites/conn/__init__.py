"""Information-based connectivity metrics and utility functions.

This submodule contains two types of functions :

1. **Connectivity metrics :** methods to estimate either the undirected or
   directed connectivity. Some methods are performed within-trials and others
   across-trials. In the case of within-trials metrics, it is then possible to
   estimate if the connectivity is modulated by the task by passing the
   connectivity arrays to `frites.workflow.WfMi`
2. **Connectivity related utility functions :** small utility functions that
   work on connectivity arrays
"""
# connectivity input conversion
from .conn_io import conn_io  # noqa

# connectivity metrics
from .conn_covgc import conn_covgc  # noqa
from .conn_dfc import conn_dfc  # noqa
from .conn_transfer_entropy import conn_transfer_entropy  # noqa

# connectivity utility functions
from .conn_sliding_windows import define_windows, plot_windows  # noqa
from .conn_utils import (conn_get_pairs, conn_reshape_undirected,  # noqa
                         conn_reshape_directed, conn_ravel_directed)
