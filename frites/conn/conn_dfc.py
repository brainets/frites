"""Dynamic Functional Connectivity."""
import numpy as np
import xarray as xr

from frites.io import set_log_level, logger

from frites.core import mi_nd_gg, copnorm_nd
from frites.config import CONFIG



def conn_dfc(data, times, roi, win_sample, verbose=None):
    """Compute the Dynamic Functional Connectivity using the GCMI.

    This function computes the Dynamic Functional Connectivity (DFC) using the
    Gaussian Copula Mutual Information (GCMI). The DFC is computed across time
    points for each trial. Note that the DFC can either be computed on windows
    manually defined or on sliding windows.

    Parameters
    ----------
    data : array_like
        Electrophysiological data array of a single subject organized as
        (n_epochs, n_roi, n_times)
    times : array_like
        Time vector array of shape (n_times,)
    roi : array_like
        ROI names of a single subject
    win_sample : array_like
        Array of shape (n_windows, 2) describing where each window start and
        finish. You can use the function :func:`frites.utils.define_windows`
        to define either manually either sliding windows.

    Returns
    -------
    dfc : array_like
        The DFC array of shape (n_epochs, n_pairs, n_windows)
    pairs : array_like
        Array of pairs of shape (n_pairs, 2)
    roi_p : array_like
        Array of shape (n_pairs,) describing the name of each pair

    See also
    --------
    define_windows, covgc
    """
    set_log_level(verbose)
    # -------------------------------------------------------------------------
    # data checking
    assert isinstance(data, np.ndarray) and (data.ndim == 3)
    n_epochs, n_roi, n_pts = data.shape
    assert (len(roi) == n_roi) and (len(times) == n_pts)
    assert isinstance(win_sample, np.ndarray) and (win_sample.ndim == 2)
    assert win_sample.dtype in CONFIG['INT_DTYPE']
    n_win = win_sample.shape[0]
    # get the non-directed pairs
    x_s, x_t = np.triu_indices(n_roi, k=1)
    n_pairs = len(x_s)
    pairs = np.c_[x_s, x_t]
    # build roi pairs names
    roi_p = [f"{roi[s]}-{roi[t]}" for s, t in zip(x_s, x_t)]

    # -------------------------------------------------------------------------
    # compute dfc
    logger.info(f'Computing DFC between {n_pairs} pairs')
    dfc = np.zeros((n_epochs, n_pairs, n_win), dtype=np.float32)
    for n_w, w in enumerate(win_sample):
        # select the data in the window and copnorm across time points
        data_w = copnorm_nd(data[..., w[0]:w[1]], axis=2)
        # compute mi between pairs
        for n_p, (s, t) in enumerate(zip(x_s, x_t)):
            dfc[:, n_p, n_w] = mi_nd_gg(data_w[:, [s], :], data_w[:, [t], :],
                                        **CONFIG["KW_GCMI"])

    # -------------------------------------------------------------------------
    # dataarray conversion
    trials = np.arange(n_epochs)
    win_times = times[win_sample]
    dfc = xr.DataArray(dfc, dims=('trials', 'roi', 'times'),
                       coords=(trials, roi_p, win_times.mean(1)))
    # add the windows used in the attributes
    dfc.attrs['win_sample'] = np.r_[tuple(win_sample)]
    dfc.attrs['win_times'] = np.r_[tuple(win_times)]

    return dfc, pairs, roi_p