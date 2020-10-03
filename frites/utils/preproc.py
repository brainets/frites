"""Pre- and post-processing functions."""
import logging

import numpy as np
import xarray as xr
from scipy.signal import savgol_filter as savgol

from frites.io import set_log_level

logger = logging.getLogger("frites")


def savgol_filter(x, h_freq, axis=None, sfreq=None, polyorder=5, verbose=None):
    set_log_level(verbose)
    # inputs checking
    if isinstance(x, xr.DataArray):
        dims = list(x.dims)
        # get axis name
        if axis is None:
            axis = 'times'
        if isinstance(axis, str):
            axis = list(x.dims).index(axis)
        # get sfreq if possible
        if not isinstance(sfreq, (int, float)):
            assert 'times' in dims
            sfreq = 1. / (x['times'].data[1] - x['times'].data[0])
    assert isinstance(h_freq, (int, float))
    assert isinstance(axis, int)
    assert isinstance(sfreq, (int, float))
    if h_freq >= sfreq / 2.:
        raise ValueError('h_freq must be less than half the sample rate')

    # get window length
    window_length = (int(np.round(sfreq / h_freq)) // 2) * 2 + 1
    logger.info(f'    Using savgol length {window_length}')

    # apply savgol depending on input type
    if isinstance(x, xr.DataArray):
        x.data = savgol(x.data, axis=axis, polyorder=polyorder,
                        window_length=window_length)
        return x
    else:
        return savgol(x, axis=axis, polyorder=polyorder,
                      window_length=window_length)
