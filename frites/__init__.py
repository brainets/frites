"""
Frites
======

Framework of Information Theory for Electrophysiological data and Statistics
"""
import logging

from frites import (io, core, conn, stats, utils, workflow, simulations,  # noqa
                    estimator)

__version__ = "0.4.0"

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
