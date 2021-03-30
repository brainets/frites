"""Dynamic Functional Connectivity."""
import numpy as np
import pandas as pd
import xarray as xr

from frites.io import set_log_level, logger
from frites.core import mi_nd_gg, copnorm_nd
from frites.config import CONFIG
from frites.utils import parallel_func
from frites.dataset import SubjectEphy


def conn_dfc(data, win_sample=None, times=None, roi=None, n_jobs=1, gcrn=True,
             agg_ch=False, verbose=None):
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
    agg_ch : bool | False
        In case there are multiple electrodes, channels, contacts or sources
        inside a brain region, specify how the data has to be aggregated. Use
        either :

            * agg_ch=False : compute the pairwise DFC aross all possible pairs
            * agg_ch=True : compute the multivariate MI

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
    if agg_ch:
        logger.info('    Grouping pairs of brain regions')
        gp = pd.DataFrame({'roi': roi}).groupby('roi').groups
        roi_gp = np.array(list(gp.keys()))
        roi_idx = np.array(list(gp.values()))
    else:
        roi_gp, roi_idx = roi, np.arange(len(roi)).reshape(-1, 1)
    n_roi = len(roi_gp)
    x_s, x_t = np.triu_indices(n_roi, k=1)
    n_pairs = len(x_s)
    # build names of pairs of brain regions
    roi_s, roi_t = roi_gp[x_s], roi_gp[x_t]
    roi_s, roi_t = np.sort(np.c_[roi_s, roi_t], axis=1).T
    roi_p = [f"{s}-{t}" for s, t in zip(roi_s, roi_t)]

    # -------------------------------------------------------------------------
    # prepare outputs and elements
    n_jobs = 1 if n_win == 1 else n_jobs
    parallel, p_fun = parallel_func(_conn_dfc, n_jobs=n_jobs, verbose=verbose,
                                    total=n_win, mesg='Estimating DFC')

    logger.info(f'Computing DFC between {n_pairs} pairs (gcrn={gcrn})')
    dfc = np.zeros((n_trials, n_pairs, n_win), dtype=np.float64)

    # -------------------------------------------------------------------------
    # compute distance correlation

    dfc = parallel(p_fun(
        x[:, :, w[0]:w[1]], x_s, x_t, roi_idx, gcrn) for w in win_sample)
    dfc = np.stack(dfc, 2)

    # -------------------------------------------------------------------------
    # dataarray conversion
    win_times = times[win_sample]
    dfc = xr.DataArray(dfc, dims=('trials', 'roi', 'times'), name='dfc',
                       coords=(trials, roi_p, win_times.mean(1)))
    # add the windows used in the attributes
    cfg = dict(
        win_sample=np.r_[tuple(win_sample)], win_times=np.r_[tuple(win_times)],
        agg_ch=agg_ch, type='dfc')
    dfc.attrs = {**cfg, **attrs}

    return dfc


def _conn_dfc(x_w, x_s, x_t, roi_idx, gcrn):
    """Parallel function for computing DFC."""
    dfc = np.zeros((x_w.shape[0], len(x_s)))
    # copnorm data only once
    if gcrn:
        x_w = copnorm_nd(x_w, axis=2)
    # compute dfc
    for n_p, (s, t) in enumerate(zip(x_s, x_t)):
        # select sources and targets time-series
        _x_s = x_w[:, roi_idx[s], :]
        _x_t = x_w[:, roi_idx[t], :]
        # compute mi between time-series
        dfc[:, n_p] = mi_nd_gg(_x_s, _x_t, traxis=-1, mvaxis=-2,
                               shape_checking=False)
    return dfc
