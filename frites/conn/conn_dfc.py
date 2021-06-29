"""Dynamic Functional Connectivity."""
import numpy as np
import xarray as xr

from frites.conn import conn_io
from frites.io import logger
from frites.estimator import GCMIEstimator
from frites.utils import parallel_func


def conn_dfc(data, win_sample=None, times=None, roi=None, agg_ch=False,
             estimator=None, gcrn=False, n_jobs=1, verbose=None):
    """Single trial Dynamic Functional Connectivity.

    This function computes the pairwise Dynamic Functional Connectivity (DFC)
    by estimating the statistical dependencies between time-series (possibly on
    sliding windows) and at the single-trial level using a measure of
    information. By default, if no estimator are provided the information
    shared between the time-series of two brain regions is estimated using the
    Gaussian-Copula Mutual Information (GCMI). However, other estimators can be
    provided (e.g correlation, distance correlation etc.)

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
    agg_ch : bool | False
        In case there are multiple electrodes, channels, contacts or sources
        inside a brain region, specify how the data has to be aggregated. Use
        either :

            * agg_ch=False : compute the pairwise DFC aross all possible pairs
            * agg_ch=True : compute the multivariate information

        Note that feature is only available for measures of information
        supporting multivariate computations.
    estimator : frites.estimator | None
        Estimator in order to measure the amount of information shared between
        two time-series coming from two distinct brain regions. Note that if
        you want to privide an estimator, be sure that it is made for
        continuous variables (mi_type='cc'). By default the Gaussian-Copula
        mutual-information is used.
    n_jobs : int | 1
        Number of jobs to use for parallel computing (use -1 to use all
        jobs). The parallel loop is set at the pair level.

    Returns
    -------
    dfc : array_like
        The DFC array of shape (n_epochs, n_pairs, n_windows)

    See also
    --------
    define_windows, conn_covgc
    """
    # ________________________________ INPUTS _________________________________
    # inputs conversion
    data, cfg = conn_io(
        data, times=times, roi=roi, agg_ch=agg_ch, win_sample=win_sample,
        pairs=None, sort=True, name='DFC', verbose=verbose,
    )

    # extract variables
    x, trials, attrs = data.data, data['y'].data, cfg['attrs']
    win_sample, win_times = cfg['win_sample'], cfg['win_times']
    x_s, x_t = cfg['x_s'], cfg['x_t']
    roi_p, roi_idx = cfg['roi_p'], cfg['roi_idx']
    n_pairs = len(x_s)

    # estimator
    if estimator is None:
        estimator = GCMIEstimator(
            mi_type='cc', copnorm=False, biascorrect=False, demeaned=False,
            verbose=verbose)
    assert estimator.settings['mi_type'] == 'cc', (
        "Estimator should extract information between two continuous "
        "variables (mi_type='cc')")
    fcn = estimator.get_function()

    # __________________________________ DFC __________________________________
    # function to put in parallel

    def para_dfc(i_s, i_t):
        dfc = np.zeros((len(trials), 1, len(win_sample)))
        x_s, x_t = x[:, roi_idx[i_s], :], x[:, roi_idx[i_t], :]
        for n_w, (w_s, w_e) in enumerate(win_sample):
            dfc[:, 0, n_w] = fcn(x_s[..., w_s:w_e], x_t[..., w_s:w_e])
        return dfc

    # prepare parallel function
    n_jobs = 1 if n_pairs == 1 else n_jobs
    parallel, p_fun = parallel_func(para_dfc, n_jobs=n_jobs, verbose=verbose,
                                    total=n_pairs, mesg='Estimating DFC')

    logger.info(f'Computing DFC between {n_pairs} pairs (gcrn={gcrn})')

    # compute dfc
    dfc = parallel(p_fun(i_s, i_t) for i_s, i_t in zip(x_s, x_t))
    dfc = np.concatenate(dfc, axis=1)

    # ________________________________ OUTPUTS ________________________________
    # dataarray conversion
    dfc = xr.DataArray(dfc, dims=('trials', 'roi', 'times'),
                       coords=(trials, roi_p, win_times),
                       name=f'DFC ({estimator.name})')

    # add the windows used in the attributes
    cfg = dict(
        win_sample=np.r_[tuple(win_sample)], win_times=np.r_[tuple(win_times)],
        agg_ch=str(agg_ch), type='dfc', estimator=estimator.name)
    dfc.attrs = {**cfg, **attrs}

    return dfc
