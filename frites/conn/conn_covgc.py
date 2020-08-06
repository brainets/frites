"""Single-trial covariance-based Granger Causality for gaussian variables."""
import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from frites.io import set_log_level, logger
from frites.config import CONFIG
from frites.core.gcmi_nd import cmi_nd_ggg
from frites.core.copnorm import copnorm_nd



###############################################################################
###############################################################################
#                                COVGC ENTROPY
###############################################################################
###############################################################################


LOG2 = np.log(2)


def entr(xy):
    """Entropy of a gaussian variable.

    This function computes the entropy of a gaussian variable for a 2D input.
    """
    # manually compute the covariance (faster)
    n_r, n_c = xy.shape
    xy = xy - xy.mean(axis=1, keepdims=True)
    out = np.empty((n_r, n_r), xy.dtype, order='C')
    np.dot(xy, xy.T, out=out)
    out /= (n_c - 1)
    # compute determinant
    det = np.linalg.det(out)
    if not det > 0:
        raise ValueError(f"Can't estimate the entropy properly of the input "
                         f"matrix of shape {xy.shape}. Try to increase the "
                         "step")
    h = np.log(det)
    return h


def _covgc(d_s, d_t, ind_tx, t0):
    """Compute the covGC for a single pair.

    This function computes the covGC for a single pair, across multiple trials,
    at different time indices.
    """
    n_trials, n_times = d_s.shape[0], len(t0)
    gc = np.empty((n_trials, n_times, 3), dtype=d_s.dtype, order='C')
    for n_ti, ti in enumerate(t0):
        # force starting indices at t0 + force row-major slicing
        ind_t0 = np.ascontiguousarray(ind_tx + ti)
        for n_tr in range(n_trials):
            # trial + temporal selection (should still be contiguous) with
            # shapes of (lag + 1, dt)
            x = d_s[n_tr, ind_t0]
            y = d_t[n_tr, ind_t0]

            # -----------------------------------------------------------------
            # Conditional Entropies
            # -----------------------------------------------------------------
            # H(Y_i+1|Y_i) = H(Y_i+1) - H(Y_i)
            det_yi1 = entr(y)
            det_yi = entr(y[1:, :])
            hycy = det_yi1 - det_yi

            # H(X_i+1|X_i) = H(X_i+1) - H(X_i)
            det_xi1 = entr(x)
            det_xi = entr(x[1:, :])
            hxcx = det_xi1 - det_xi

            # H(Y_i+1|X_i,Y_i) = H(Y_i+1,X_i,Y_i) - H(X_i,Y_i)
            det_yi1xi = entr(np.concatenate((y, x[1:, :]), axis=0))
            det_yxi = entr(np.concatenate((y[1:, :], x[1:, :]), axis=0))
            hycx = det_yi1xi - det_yxi

            # H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)
            det_xi1yi = entr(np.concatenate((x, y[1:, :]), axis=0))
            hxcy = det_xi1yi - det_yxi

            # H(X_i+1,Y_i+1|X_i,Y_i) = H(X_i+1,Y_i+1,X_i,Y_i) - H(X_i,Y_i)
            det_xi1yi1 = entr(np.concatenate((x, y), axis=0))
            hxxcyy = det_xi1yi1 - det_yxi

            # -----------------------------------------------------------------
            # Granger Causality measures
            # -----------------------------------------------------------------
            # gc(pairs(:,1) -> pairs(:,2))
            gc[n_tr, n_ti, 0] = hycy - hycx
            # gc(pairs(:,2) -> pairs(:,1))
            gc[n_tr, n_ti, 1] = hxcx - hxcy
            # gc(pairs(:,2) . pairs(:,1))
            gc[n_tr, n_ti, 2] = hycx + hxcy - hxxcyy

    return gc / (2. * LOG2)


###############################################################################
###############################################################################
#                             GAUSSIAN COPULA COVGC
###############################################################################
###############################################################################


def _gccovgc(d_s, d_t, ind_tx, t0):
    """Compute the Gaussian-Copula based covGC for a single pair.

    This function computes the covGC for a single pair, across multiple trials,
    at different time indices.
    """
    kw = CONFIG["KW_GCMI"]
    n_trials, n_times = d_s.shape[0], len(t0)
    gc = np.empty((n_trials, n_times, 3), dtype=d_s.dtype, order='C')
    for n_ti, ti in enumerate(t0):
        # force starting indices at t0 + force row-major slicing
        ind_t0 = np.ascontiguousarray(ind_tx + ti)
        x = d_s[:, ind_t0]
        y = d_t[:, ind_t0]
        # temporal selection
        x_pres, x_past = x[:, [0], :], x[:, 1:, :]
        y_pres, y_past = y[:, [0], :], y[:, 1:, :]
        xy_past = np.concatenate((x[:, 1:, :], y[:, 1:, :]), axis=1)
        # copnorm over the last axis (avoid copnorming several times)
        x_pres = copnorm_nd(x_pres, axis=-1)
        x_past = copnorm_nd(x_past, axis=-1)
        y_pres = copnorm_nd(y_pres, axis=-1)
        y_past = copnorm_nd(y_past, axis=-1)
        xy_past = copnorm_nd(xy_past, axis=-1)

        # -----------------------------------------------------------------
        # Granger Causality measures
        # -----------------------------------------------------------------
        # gc(pairs(:,1) -> pairs(:,2))
        gc[:, n_ti, 0] = cmi_nd_ggg(y_pres, x_past, y_past, **kw)
        # gc(pairs(:,2) -> pairs(:,1))
        gc[:, n_ti, 1] = cmi_nd_ggg(x_pres, y_past, x_past, **kw)
        # gc(pairs(:,2) . pairs(:,1))
        gc[:, n_ti, 2] = cmi_nd_ggg(x_pres, y_pres, xy_past, **kw)
    
    return gc



###############################################################################
###############################################################################
#                         CONDITIONAL GAUSSIAN COPULA COVGC
###############################################################################
###############################################################################


def _cond_gccovgc(data, s, t, ind_tx, t0, conditional=True):
    """Compute the Gaussian-Copula based covGC for a single pair.

    This function computes the covGC for a single pair, across multiple trials,
    at different time indices.
    """
    conditional = conditional if data.shape[1] > 2 else False
    kw = CONFIG["KW_GCMI"]
    d_s, d_t = data[:, s, :], data[:, t, :]
    n_lags, n_dt = ind_tx.shape
    n_trials, n_times = d_s.shape[0], len(t0)
    gc = np.empty((n_trials, n_times, 3), dtype=d_s.dtype, order='C')
    # define z past
    roi_range = np.array([k for k in range(data.shape[1]) if k not in [s, t]])
    z_roi = data[:, roi_range, :]  # other roi selection
    rsh = int(len(roi_range) * (n_lags - 1))
    for n_ti, ti in enumerate(t0):
        # force starting indices at t0 + force row-major slicing
        ind_t0 = np.ascontiguousarray(ind_tx + ti)
        x = d_s[:, ind_t0]
        y = d_t[:, ind_t0]
        # temporal selection
        x_pres, x_past = x[:, [0], :], x[:, 1:, :]
        y_pres, y_past = y[:, [0], :], y[:, 1:, :]
        xy_past = np.concatenate((x[:, 1:, :], y[:, 1:, :]), axis=1)
        # conditional granger causality case
        if conditional:
            # condition by the past of every other possible sources 
            z_past = z_roi[..., ind_t0[1:, :]]  # (lag_past, dt) selection
            z_past = z_past.reshape(n_trials, rsh, n_dt)
            # cat with past
            yz_past = np.concatenate((y_past, z_past), axis=1)
            xz_past = np.concatenate((x_past, z_past), axis=1)
            xyz_past = np.concatenate((xy_past, z_past), axis=1)
        else:
            yz_past, xz_past, xyz_past = y_past, x_past, xy_past
        # copnorm over the last axis (avoid copnorming several times)
        x_pres = copnorm_nd(x_pres, axis=-1)
        x_past = copnorm_nd(x_past, axis=-1)
        y_pres = copnorm_nd(y_pres, axis=-1)
        y_past = copnorm_nd(y_past, axis=-1)
        yz_past = copnorm_nd(yz_past, axis=-1)
        xz_past = copnorm_nd(xz_past, axis=-1)
        xyz_past = copnorm_nd(xyz_past, axis=-1)

        # -----------------------------------------------------------------
        # Granger Causality measures
        # -----------------------------------------------------------------
        # gc(pairs(:,1) -> pairs(:,2))
        gc[:, n_ti, 0] = cmi_nd_ggg(y_pres, x_past, yz_past, **kw)
        # gc(pairs(:,2) -> pairs(:,1))
        gc[:, n_ti, 1] = cmi_nd_ggg(x_pres, y_past, xz_past, **kw)
        # gc(pairs(:,2) . pairs(:,1))
        gc[:, n_ti, 2] = cmi_nd_ggg(x_pres, y_pres, xyz_past, **kw)
    
    return gc


###############################################################################
###############################################################################
#                            HIGH-LEVEL CONN_COVGC
###############################################################################
###############################################################################


def conn_covgc(data, dt, lag, t0, step=1, roi=None, times=None, method='gc',
               conditional=False, n_jobs=-1, verbose=None):
    r"""Single-trial covariance-based Granger Causality for gaussian variables.

    This function computes the (conditional) covariance-based Granger Causality
    (covgc) for each trial.

    .. note::
        **Total Granger interdependence**

            * TGI = gc.sum(axis=-1) = gc(x->y) + gc(y->x) + gc(x.y)
            * TGI = Hycy + Hxcx - Hxxcyy

        **Relations between Mutual Informarion and conditional entropies**

        This quantity can be defined as the Increment of Total Interdependence
        and it can be calculated from the different of two mutual informations
        as follows

        .. math::

            Ixxyy  &=  I(X_{i+1}, X_{i}|Y_{i+1}, Y_{i}) \\
                   &=  H(X_{i+1}) + H(Y_{i+1}) - H(X_{i+1},Y_{i+1}) \\
                   &=  log(det_{xi1}) + log(det_{yi1}) - log(det_{xyi1}) \\
            Ixy    &=  I(X_{i}|Y_{i}) \\
                   &=  H(X_{i}) + H(Y_{i}) - H(X_{i}, Y_{i}) \\
                   &=  log(det_{xi}) + log(det_{yi}) - log(det_{yxi}) \\
            ITI    &= Ixxyy - Ixy

    Parameters
    ----------
    data : array_like
        Electrophysiological data array of a single subject organized as
        (n_epochs, n_roi, n_times)
    dt : int
        Duration of the time window for covariance correlation in samples
    lag : int
        Number of samples for the lag within each trial
    t0 : array_like
        Array of zero time in samples of length (n_window,)
    step : int | 1
        Number of samples stepping in the past for the lag within each trial
    roi : array_like | None
        Array of ROI names of shape (n_roi,). If None, default ROI names will
        be used instead
    times : array_like | None
        Time vector of shape (n_times,). If None, a default vector will be used
        instead
    method : {'gauss', 'gc'}
        Method for the estimation of the covgc. Use either 'gauss' which
        assumes that the time-points are normally distributed or 'gc' in order
        to use the gaussian-copula.
    conditional : bool | False
        If True, the conditional Granger Causality is computed i.e the past is
        also conditioned by the past of other sources.
    n_jobs : int | -1
        Number of jobs to use for parallel computing (use -1 to use all
        jobs). The parallel loop is set at the pair level.

    Returns
    -------
    gc : array_like
        Granger Causality arranged as (n_epochs, n_pairs, n_windows, 3) where
        the last dimension means :

            * 0 : pairs[:, 0] -> pairs[:, 1] (x->y)
            * 1 : pairs[:, 1] -> pairs[:, 0] (y->x)
            * 2 : instantaneous  (x.y)
    pairs : array_like
        Array of pairs of shape (n_pairs, 2)

    References
    ----------
    Brovelli et al., 2015 :cite:`brovelli2015characterization`

    See also
    --------
    conn_dfc
    """
    set_log_level(verbose)
    # -------------------------------------------------------------------------
    # input checking
    if isinstance(t0, CONFIG['INT_DTYPE']) or isinstance(
        t0, CONFIG['FLOAT_DTYPE']):
        t0 = np.array([t0])
    t0 = np.asarray(t0).astype(int)
    dt, lag, step, trials = int(dt), int(lag), int(step), None
    # handle dataarray input
    if isinstance(data, xr.DataArray):
        if isinstance(roi, str):
            roi = data[roi].data
        if isinstance(times, str):
            times = data[times].data
        trials = data['trials'].data
        data = data.data
    # force C contiguous array because operations on row-major
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    n_epochs, n_roi, n_times = data.shape
    if trials is None:
        trials = np.arange(n_epochs)
    # default roi vector
    if roi is None:
        roi = np.array([f"roi_{k}" for k in range(n_roi)])
    roi = np.asarray(roi)
    # default time vector
    if times is None:
        times = np.arange(n_times)
    times = np.asarray(times)
    assert (len(roi) == n_roi) and (len(times) == n_times)
    # method checking
    assert method in ['gauss', 'gc']
    fcn = dict(gauss=_covgc, gc=_gccovgc)[method]

    # -------------------------------------------------------------------------
    # build generic time indices (just need to add t0 to it)
    rows, cols = np.mgrid[0:lag + 1, 0:dt]
    # step in the past lags
    rows = rows[::step, :]
    cols = cols[::step, :]
    # create index for all lags and timespoints
    ind_tx = cols - rows
    # build output time vector
    times_p = np.empty((len(t0)), dtype=times.dtype, order='C')
    for n_t, t in enumerate(t0):
        times_p[n_t] = times[ind_tx[0, :] + t].mean()
    # get the non-directed pairs and build roi pairs names
    x_s, x_t = np.triu_indices(n_roi, k=1)
    pairs = np.c_[x_s, x_t]
    roi_p = np.array([f"{roi[s]}-{roi[t]}" for s, t in zip(x_s, x_t)])
    # check the ratio between lag and dt
    ratio = 100 * (ind_tx.shape[0] / (step * ind_tx.shape[1]))
    if not 10. <= ratio <= 15.:
        _step = int(np.ceil((lag + 1) / (.15 * dt)))
        logger.warning(f"The ratio between the lag and dt is {ratio}%. It's "
                       f"recommended to conserve this ratio between 10-15%."
                       f" Try with a step={_step}")
    logger.debug(f"Index shape : {ind_tx.shape}")

    # -------------------------------------------------------------------------
    ext = 'conditional' if conditional else ''
    # compute covgc and parallel over pairs
    logger.info(f"Compute the {ext} covgc (method={method}, n_pairs={len(x_s)}"
                f"; n_windows={len(t0)}, lag={lag}, dt={dt}, step={step})")
    if not conditional:
        gc = Parallel(n_jobs=n_jobs)(delayed(fcn)(
            data[:, s, :], data[:, t, :], ind_tx, t0) for s, t in zip(
            x_s, x_t))
    else:
        gc = Parallel(n_jobs=n_jobs)(delayed(_cond_gccovgc)(
                data, s, t, ind_tx, t0) for s, t in zip(x_s, x_t))
    gc = np.stack(gc, axis=1)

    # -------------------------------------------------------------------------
    # change output type
    dire = np.array(['x->y', 'y->x', 'x.y'])
    gc = xr.DataArray(gc, dims=('trials', 'roi', 'times', 'direction'),
                      coords=(trials, roi_p, times_p, dire))
    # set attributes
    gc.attrs['lag'] = lag
    gc.attrs['step'] = step
    gc.attrs['dt'] = dt
    gc.attrs['t0'] = t0
    gc.attrs['conditional'] = conditional

    return gc, pairs, roi_p, times_p


if __name__ == '__main__':
    from frites.simulations import StimSpecAR
    import matplotlib.pyplot as plt

    ss = StimSpecAR()
    ar = ss.fit(ar_type='ding_3', n_stim=2, n_epochs=20)
    # plot the model
    # plt.figure(figsize=(7, 8))
    # ss.plot()
    # compute covgc
    dt, lag, step = 50, 5, 2
    t0 = np.arange(lag, ar.shape[-1] - dt, step)
    gc = conn_covgc(ar, roi='roi', times='times', dt=dt, lag=lag, t0=t0,
                    n_jobs=-1, conditional=False)[0]
    ss.plot_covgc(gc=gc)
    plt.show()
