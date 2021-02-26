"""Dynamic Functional Connectivity."""
import numpy as np
import pandas as pd
import xarray as xr

from frites.io import set_log_level, logger
from frites.core import mi_1d_gg, copnorm_nd
from frites.config import CONFIG
from frites.utils import parallel_func
from frites.dataset import SubjectEphy

from mne.utils import ProgressBar



def conn_dfc(data, win_sample=None, times=None, roi=None, n_jobs=1, gcrn=True,
             verbose=None):
    """Single trial Dynamic Functional Connectivity.

    This function computes the Dynamic Functional Connectivity (DFC) using the
    Gaussian Copula Mutual Information (GCMI). The DFC is computed across time
    points for each trial. Note that the DFC can either be computed on windows
    manually defined or on sliding windows.

    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported :

            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)

    win_sample : array_like | None
        Array of shape (n_windows, 2) describing where each window start and
        finish. You can use the function :func:`frites.conn.define_windows`
        to define either manually either sliding windows. If None, the entire
        time window is used instead.
    times : array_like | None
        Time vector array of shape (n_times,). If the input is an xarray, the
        name of the time dimension can be provided
    roi : array_like | None
        ROI names of a single subject. If the input is an xarray, the
        name of the ROI dimension can be provided
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
    # inputs conversion and data checking
    set_log_level(verbose)
    if isinstance(data, xr.DataArray):
        trials, attrs = data[data.dims[0]].data, data.attrs
    else:
        trials, attrs = np.arange(data.shape[0]), {}
    # internal conversion
    data = SubjectEphy(data, y=trials, roi=roi, times=times)
    x, roi, times = data.data, data['roi'].data, data['times'].data
    trials = data['y'].data
    n_trials = len(trials)
    # deal with the win_sample array
    if win_sample is None:
        win_sample = np.array([[0, len(times) - 1]])
    assert isinstance(win_sample, np.ndarray) and (win_sample.ndim == 2)
    assert win_sample.dtype in CONFIG['INT_DTYPE']
    n_win = win_sample.shape[0]

    # -------------------------------------------------------------------------
    # find group of brain regions
    gp = pd.DataFrame({'roi': roi}).groupby('roi').groups
    roi_gp, roi_idx = list(gp.keys()), list(gp.values())
    n_roi = len(roi_gp)
    x_s, x_t = np.triu_indices(n_roi, k=1)
    n_pairs = len(x_s)
    pairs = np.c_[x_s, x_t]
    roi_p = [f"{roi_gp[s]}-{roi_gp[t]}" for s, t in zip(x_s, x_t)]

    # -------------------------------------------------------------------------
    # prepare outputs and elements
    parallel, p_fun = parallel_func(mi_1d_gg, n_jobs=n_jobs, verbose=verbose)
    pbar = ProgressBar(range(n_pairs), mesg='Estimating DFC')

    logger.info(f'Computing DFC between {n_pairs} pairs (gcrn={gcrn})')
    dfc = np.zeros((n_trials, n_pairs, n_win), dtype=np.float64)
    q = 0

    # -------------------------------------------------------------------------
    # compute distance correlation
    for s, t in zip(x_s, x_t):
        for n_w, w in enumerate(win_sample):
            _x_s = x[:, roi_idx[s], w[0]:w[1]]
            _x_t = x[:, roi_idx[t], w[0]:w[1]]
            if gcrn:
                _x_s = copnorm_nd(_x_s, axis=2)
                _x_t = copnorm_nd(_x_t, axis=2)
            _dfc = parallel(p_fun(
                _x_s[tr, ...], _x_t[tr, ...]) for tr in range(n_trials))
            dfc[:, q, n_w] = np.array(_dfc)
        q += 1
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
