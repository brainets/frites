"""Dynamic Functional Connectivity."""
import numpy as np
import xarray as xr

from frites.io import set_log_level, logger
from frites.core import mi_nd_gg, copnorm_nd
from frites.config import CONFIG
from frites.utils import parallel_func

from frites.conn.conn_io import conn_io

from mne.utils import ProgressBar



def conn_dfc(data, win_sample, times=None, roi=None, n_jobs=1, gcrn=True,
             verbose=None):
    """Single trial Dynamic Functional Connectivity.

    This function computes the Dynamic Functional Connectivity (DFC) using the
    Gaussian Copula Mutual Information (GCMI). The DFC is computed across time
    points for each trial. Note that the DFC can either be computed on windows
    manually defined or on sliding windows.

    Parameters
    ----------
    data : array_like
        Electrophysiological data array of a single subject organized as
        (n_epochs, n_roi, n_times)
    win_sample : array_like
        Array of shape (n_windows, 2) describing where each window start and
        finish. You can use the function :func:`frites.conn.define_windows`
        to define either manually either sliding windows.
    times : array_like | None
        Time vector array of shape (n_times,)
    roi : array_like | None
        ROI names of a single subject
    n_jobs : int | 1
        Number of jobs to use for parallel computing (use -1 to use all
        jobs). The parallel loop is set at the pair level.
    gcrn : bool | True
        Specify if the Gaussian Copula Rank Normalization should be applied.
        If the data are normalized (e.g z-score) this parameter can be set to
        False because the data can be considered as gaussian over time.

    Returns
    -------
    dfc : array_like
        The DFC array of shape (n_epochs, n_pairs, n_windows)

    See also
    --------
    define_windows, conn_covgc
    """
    set_log_level(verbose)
    # -------------------------------------------------------------------------
    # inputs conversion
    data, trials, roi, times, attrs = conn_io(data, roi=roi, times=times,
                                              verbose=verbose)

    # -------------------------------------------------------------------------
    # data checking
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
    logger.info(f'Computing DFC between {n_pairs} pairs (gcrn={gcrn})')
    # get the parallel function
    parallel, p_fun = parallel_func(mi_nd_gg, n_jobs=n_jobs, verbose=verbose,
                                    prefer='threads')
    pbar = ProgressBar(range(n_win), mesg='Estimating DFC')

    dfc = np.zeros((n_epochs, n_pairs, n_win), dtype=np.float32)
    with parallel as para:
        for n_w, w in enumerate(win_sample):
            # select the data in the window and copnorm across time points
            data_w = data[..., w[0]:w[1]]
            # apply gcrn over time
            if gcrn:
                data_w = copnorm_nd(data_w, axis=2)
            # compute mi between pairs
            _dfc = para(
                p_fun(data_w[:, [s], :], data_w[:, [t], :],
                      **CONFIG["KW_GCMI"]) for s, t in zip(x_s, x_t))
            dfc[..., n_w] = np.stack(_dfc, axis=1)
            pbar.update_with_increment_value(1)

    # -------------------------------------------------------------------------
    # dataarray conversion
    win_times = times[win_sample]
    dfc = xr.DataArray(dfc, dims=('trials', 'roi', 'times'), name='dfc',
                       coords=(trials, roi_p, win_times.mean(1)))
    # add the windows used in the attributes
    cfg = dict(win_sample=np.r_[tuple(win_sample)],
               win_times=np.r_[tuple(win_times)], type='dfc')
    dfc.attrs = {**cfg, **attrs}

    return dfc
