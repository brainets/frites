"""Single-trial covariance-based Granger Causality for gaussian variables."""
import numpy as np
from joblib import Parallel, delayed

from frites.io import set_log_level, logger
from frites.config import CONFIG



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
    h = np.log(np.linalg.det(out))
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
            # trial + temporal selection (should still be contiguous)
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
            # gc(pairs(:,1)->pairs(:,2))
            gc[n_tr, n_ti, 0] = hycy - hycx
            # gc(pairs(:,2)->pairs(:,1))
            gc[n_tr, n_ti, 1] = hxcx - hxcy
            # gc(x.y)
            gc[n_tr, n_ti, 2] = hycx + hxcy - hxxcyy

    return gc


def covgc(data, dt, lag, t0, roi=None, times=None, output_type='array',
          verbose=None, n_jobs=-1):
    r"""Single-trial covariance-based Granger Causality for gaussian variables.

    This function computes the covariance-based Granger Causality (covgc) for
    each trial.

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
    roi : array_like | None
        Array of ROI names of shape (n_roi,). If None, default ROI names will
        be used instead
    times : array_like | None
        Time vector of shape (n_times,). If None, a default vector will be used
        instead
    output_type : {'array', 'dataarray'}
        Output type, either standard NumPy array or xarray.DataArray
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
    dfc_gc
    """
    set_log_level(verbose)
    # -------------------------------------------------------------------------
    # input checking
    if isinstance(t0, CONFIG['INT_DTYPE']) or isinstance(
        t0, CONFIG['FLOAT_DTYPE']):
        t0 = np.array([t0])
    dt, lag, t0 = int(dt), int(lag), np.asarray(t0).astype(int)
    # force C contiguous array because operations on row-major
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    n_epochs, n_roi, n_times = data.shape
    # default roi vector
    if roi is None:
        roi = np.array([f"roi_{k}" for k in range(n_roi)])
    roi = np.asarray(roi)
    # default time vector
    if times is None:
        times = np.arange(n_times)
    times = np.asarray(times)
    assert (len(roi) == n_roi) and (len(times) == n_times)

    # -------------------------------------------------------------------------
    # build generic time indices (just need to add t0 to it)
    rows, cols = np.mgrid[0:lag + 1, 0:dt]
    ind_tx = cols - rows
    # build output time vector
    times_p = np.empty((len(t0)), dtype=times.dtype, order='C')
    for n_t, t in enumerate(t0):
        times_p[n_t] = times[ind_tx[0, :] + t].mean()
    # get the non-directed pairs and build roi pairs names
    x_s, x_t = np.triu_indices(n_roi, k=1)
    pairs = np.c_[x_s, x_t]
    roi_p = np.array([f"{roi[s]}-{roi[t]}" for s, t in zip(x_s, x_t)])

    # -------------------------------------------------------------------------
    # compute covgc and parallel over pairs
    logger.info(f"Compute the covgc (n_pairs={len(x_s)}; n_windows={len(t0)})")
    gc = Parallel(n_jobs=n_jobs)(delayed(_covgc)(
        data[:, s, :], data[:, t, :], ind_tx, t0) for s, t in zip(x_s, x_t))
    gc = np.stack(gc, axis=1)

    # -------------------------------------------------------------------------
    # change output type
    if output_type is 'dataarray':
        from xarray import DataArray
        trials = np.arange(n_epochs)
        dire = np.array(['x->y', 'y->x', 'x.y'])
        gc = DataArray(gc, dims=('trials', 'roi', 'times', 'direction'),
                       coords=(trials, roi_p, times_p, dire))
        # set attributes
        gc.attrs['lag'] = lag
        gc.attrs['dt'] = dt
        gc.attrs['t0'] = t0

    return gc, pairs, roi_p, times_p
