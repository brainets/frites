"""
Frites
======

Framework for Information Theoretical analyses of Electrophysiological data and Statistics
"""
import logging

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
