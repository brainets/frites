"""
Frites
======

Framework of Information Theory for Electrophysiological data and Statistics
"""
import logging

from frites import io, core, stats, utils, workflow, simulations  # noqa

# Set 'info' as the default logging level
logger = logging.getLogger('frites')
io.set_log_level('info')

__version__ = "0.1.0"
