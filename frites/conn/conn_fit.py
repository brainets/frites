"""Feature specific information transfer (Numba compliant)."""
import numpy as np
import xarray as xr

from frites.utils import jit

from frites.conn import conn_io, _conn_mi
from frites.core import mi_nd_gg, mi_model_nd_gd, copnorm_nd
from frites.io import set_log_level, logger, check_attrs
from frites.config import CONFIG

from mne.utils import ProgressBar


def conn_fit(data, y, roi=None, times=None, mi_type='cc', gcrn=True,
             max_delay=.3, avg_delay=False, net=False, sfreq=None, verbose=None,
             **kw_links):
    """Feature-specific information transfer.

    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported :

            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)

    y : array_like
        The feature of shape (n_trials,). This feature vector can either be
        categorical and in that case, the mutual information type has to 'cd'
        or y can also be a continuous regressor and in that case the mutual
        information type has to be 'cc'
    roi : array_like | None
        Array of region of interest name of shape (n_roi,)
    times : array_like | None
        Array of time points of shape (n_times,)
    mi_type : {'cc', 'cd'}
        Mutual information type. Switch between :
            * 'cc' : if the y input is a continuous regressor
            * 'cd' : if the y input is a discret vector with categorical
              integers inside
    gcrn : bool | True
        Specify if the Gaussian Copula Rank Normalization should be applied.
        Default is True.
    max_delay : float | .3
        Maximum delay for past conditioning
    avg_delay : bool | False
        If False (default) the returned FIT is aggregated across delays. If
        True, the returned FIT is going to contained the additional dimension
        corresponding to the number of delays used.
    sfreq : float | None
        The sampling frequency.
    kw_links : dict | {}
        Additional arguments for selecting links to compute are passed to the
        function :func:`frites.conn.conn_links`

    Returns
    -------
    fit : array_like
        The feature specific information transfer of shape (n_pairs, n_times)
        if avg_delay is False or (n_pairs, n_delays, n_times) if avg_delay is
        True.

    See also
    --------
    conn_links
    """
    # _________________________________ INPUTS ________________________________
    # inputs conversion
    kw_links.update({'directed': True, 'net': False})
    data, cfg = conn_io(
        data, y=y, times=times, roi=roi, agg_ch=False, win_sample=None,
        name='FIT', verbose=verbose, sfreq=sfreq, kw_links=kw_links
    )

    # extract variables
    x, attrs = data.data, cfg['attrs']
    y, roi, times = data['y'].data, data['roi'].data, data['times'].data

    # indices for the souces and targets
    i_s, i_t = cfg['x_s'], cfg['x_t']
    roi_p, n_pairs = cfg['roi_p'], len(i_s)

    # delay conversion
    if isinstance(max_delay, int):
        max_delay = max_delay / cfg['sfreq']
    n_delays = int(np.round(max_delay * cfg['sfreq']))

    # build the indices when using multi-variate mi
    n_trials, n_roi, n_times = len(y), len(roi), len(times)

    logger.info(f"Compute FIT on {n_pairs} connectivity pairs "
                f"(max_delay={max_delay})")
    # gcrn
    if gcrn:
        logger.info("    Apply the Gaussian Copula Rank Normalization")
        x = copnorm_nd(x, axis=0)
        if mi_type == 'cc':
            y = copnorm_nd(y, axis=0)

    # transpose the data to be (n_roi, n_times, 1, n_trials)
    x = np.transpose(x, (1, 2, 0))[..., np.newaxis, :]

    # __________________________ MUTUAL INFORMATION ___________________________

    # compute mi between each node x (brain data) and y (task-related var)
    mi_xy = np.zeros((n_roi, n_times), dtype=float)
    for n_r in range(n_roi):
        mi_xy[n_r, :] = _conn_mi(x[n_r, :, :], y, mi_type)
    mi_xy_s = mi_xy[i_s, :]
    mi_xy_t = mi_xy[i_t, :]

    # compute mi between past and present of sources and targets
    cfg_mi = CONFIG["KW_GCMI"]
    mi_x_sptf = np.zeros((n_pairs, n_delays, n_times), dtype=float)
    mi_x_tptf = np.zeros((n_pairs, n_delays, n_times), dtype=float)

    for n_d in range(n_delays):
        # define indices
        idx_past = slice(n_d, n_d + n_times - n_delays - 1)
        idx_pres = slice(n_delays + 1, n_times)

        for n_l in range(n_pairs):
            # source past; target past; target present
            _sp = x[i_s[n_l], idx_past, :]
            _tp = x[i_t[n_l], idx_past, :]
            _tf = x[i_t[n_l], idx_pres, :]

            # I(source_{past}; target_{pres})
            mi_x_sptf[n_l, n_d, idx_pres] = mi_nd_gg(
                _sp, _tf, **cfg_mi
            ).squeeze()

            # I(target_{past}; target_{pres})
            mi_x_tptf[n_l, n_d, idx_pres] = mi_nd_gg(
                _tp, _tf, **cfg_mi
            ).squeeze()

    # __________________________________ FIT __________________________________
    # time indices for target roi
    t_start = list(range(n_delays, n_times))

    # Compute FIT on original MI values
    if avg_delay:
        fit_sh = (n_pairs, n_delays, n_times - n_delays)
    else:
        fit_sh = (n_pairs, n_times - n_delays)
    fit = np.zeros(fit_sh, dtype=np.float32)

    # I(target_pres; cue)
    mi_xy_t_pres = mi_xy_t[..., t_start]

    # I(source_past; target_pres)
    mi_x_st_pres =  mi_x_sptf[..., t_start]

    # I(target_past; target_pres) = mi_x_t
    mi_x_t_pres = mi_x_tptf[..., t_start]

    # Loop over delays for past of target and sources
    for n_d in range(n_delays):

        # Delay indices
        delays = list(range(n_d, n_times - n_delays + n_d))

        # PID with cue as target var
        # I(target_{past}; cue)
        mi_xy_t_past = mi_xy_t[..., delays]
        # I(source_{past}; cue)
        mi_xy_s_past = mi_xy_s[..., delays]

        # redundancy between sources and target about S (MMI-based)
        red_s_t = np.minimum(mi_xy_s_past, mi_xy_t_pres)
        # redundancy between sources, target present and target past about S
        red_all = np.minimum(red_s_t, mi_xy_t_past)
        # first term of FIT with the cue as target var
        fit_cue = red_s_t - red_all

        # PID with target pres as target var
        # redundancy between sources and target about target pres (MMI-based)
        red_x_t = np.minimum(mi_xy_t_pres, mi_x_st_pres[:, n_d, :])
        # redundancy between sources, target present and target past about S
        red_all = np.minimum(red_x_t, mi_x_t_pres[:, n_d, :])
        # second term of FIT with x pres as target var
        fit_t_pres = red_x_t - red_all

        if avg_delay:
            fit[:, n_d, :] = np.minimum(fit_cue, fit_t_pres)
        else:
            fit += np.minimum(fit_cue, fit_t_pres)

    # ________________________________ OUTPUTS ________________________________
    # rebuild time vector
    times = times[n_delays:]
    delay = np.arange(1, n_delays + 1)[::-1] / cfg['sfreq']

    # net transfer
    if net:
        # get unique pairs
        i_s, i_t = np.stack((i_s, i_t))[:, i_s < i_t]
        # computes net transfer
        fit = fit[i_s, :] - fit[i_t, :]
        roi_p = np.array(roi_p)[i_s]

    # xarray conversion
    if avg_delay:
        dims, coords = ('roi', 'delays', 'times'), (roi_p, delay, times)
    else:
        dims, coords = ('roi', 'times'), (roi_p, times)
    attrs['max_delay'] = max_delay
    attrs['mi_type'] = mi_type
    fit = xr.DataArray(
        fit, name='FIT', dims=dims, coords=coords, attrs=check_attrs(attrs)
    )

    return fit



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    net = False
    avg_delay = False

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # import xarray as xr
    # import numpy as np

    # n_trials = 100
    # n_roi = 3
    # n_times = 1000
    # sfreq = 512.

    # trials = np.array([0] * 50 + [1] * 50)
    # roi = [f"roi_{n_r}" for n_r in range(n_roi)]
    # times = np.arange(n_times) / sfreq

    # x = np.random.rand(n_trials, n_roi, n_times)
    # x = xr.DataArray(
    #     x, dims=('trials', 'roi', 'times'), coords=(trials, roi, times))
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    from frites.simulations import StimSpecAR

    ar_type = 'hga'
    n_stim = 3
    n_epochs = 400

    ss = StimSpecAR()
    x = ss.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim,
               random_state=0)
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    fit = conn_fit(x, y='trials', roi='roi', times='times', mi_type='cd',
                   max_delay=.3, net=net, verbose=False, avg_delay=avg_delay)
    if net:
        fit.plot(x='times')
    else:
        fit.plot(x='times', col='roi')

    plt.show()
