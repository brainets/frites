"""
Frites
======

Framework for Information Theoretical analyses of Electrophysiological data and Statistics
"""
import logging

import numpy as np

# -----------------------------------------------------------------------------
# numpy / mne transitional compatibility shim
#
# numpy >= 2.4 removed `np.in1d` (deprecated since numpy 2.0 in favor of
# `np.isin`), but mne (up to and including 1.12.1, its latest release as of
# writing) still calls `np.in1d` internally (e.g. in mne/epochs.py) when
# building Epochs. mne's development branch has already dropped that call,
# but no release with the fix is out yet. Restore the alias so mne keeps
# working until such a release is available.
# TODO: drop this once the minimum supported mne version no longer needs it.
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

from frites import (
    io, core, conn, plot, stats, utils, workflow, simulations,  estimator  # noqa
)

__version__ = "0.4.5"

# -----------------------------------------------------------------------------
# Set 'info' as the default logging level
logger = logging.getLogger('frites')
io.set_log_level('info')

# -----------------------------------------------------------------------------
# get / set config


def get_config():
    """Get the global configuration of frites."""
    from frites.config import CONFIG
    return CONFIG


def set_config(key, value, verbose=None):
    """Change the global config of frites.

    Parameters
    ----------
    key : string
        Entry of the config
    value : dict / list
        The new value for the selected key. The type should be the same as the
        default one
    """
    io.set_log_level(verbose)
    assert isinstance(key, str)
    CONFIG = get_config()  # noqa
    assert key in CONFIG.keys(), f"The key {key} doesn't exist."
    CONFIG[key] = value
    logger.info(f"The key {key} has been updated")


def set_mpl_style(style='frites'):
    """Set matplotlib style.

    Parameters
    ----------
    style : array_like
        Style name. Use either "frites" for a white background or "ggfrites"
        for a grey brackground like with ggplot.
    """
    from pkg_resources import resource_filename
    import matplotlib.pyplot as plt
    assert style in ["frites", "ggfrites"]
    path_style = resource_filename('frites', f'data/{style}.mplstyle')
    plt.style.use(path_style)
