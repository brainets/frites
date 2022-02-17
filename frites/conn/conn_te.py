"""Across-trials transfer entropy.
"""
import numpy as np
import xarray as xr

from frites.conn import conn_io
from frites.io import set_log_level, logger, check_attrs
from frites.core import cmi_nd_ggg, copnorm_nd
from frites.config import CONFIG
from frites.utils import parallel_func


def _para_te(x_s, x_t, max_delay, return_delays, delays):
    """Compute TE for a single pair of (source, target)."""
    n_pts = x_s.shape[0]
    te = np.zeros((len(delays), n_pts - max_delay), dtype=float)
    for idx_pr, n_pr in enumerate(range(max_delay, n_pts)):
        # build pas indices
        sl_past = delays + n_pr
        # past selection [present - delay, present[
        xs_past = x_s[sl_past, ...]
        xt_past = x_t[sl_past, ...]
        # present selection [present]
        xt_pres = np.tile(x_t[[n_pr], ...], (len(delays), 1, 1))
        te[:, n_pr - max_delay] = cmi_nd_ggg(
            xs_past, xt_pres, xt_past, **CONFIG["KW_GCMI"])

    if return_delays:
        return te
    else:
        return te.mean(0)


def conn_te(data, times=None, roi=None, min_delay=0, max_delay=30,
            step_delay=1, return_delays=False, gcrn=True, sfreq=None,
            n_jobs=1, verbose=None, **kw_links):
    """Compute the across-trials transfer entropy (TE).

    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported :

            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)

    times : array_like | None
        Time vector array of shape (n_times,). If the input is an xarray, the
        name of the time dimension can be provided
    roi : array_like | None
        ROI names of a single subject. If the input is an xarray, the
        name of the ROI dimension can be provided
    max_delay : int | 30
        Number of time points defining where to stop looking at in the past.
        Increasing this maximum delay input can lead to slower computations
    step_delay : int | 1
        Step between delays to test. By default, test every delays
    return_delays : bool | False
        Specify whether the returned TE should be average across delays (False)
        or not (True).
    sfreq : float | None
        Sampling frequency
    gcrn : bool | True
        Specify if the Gaussian Copula Rank Normalization should be applied.
        Default is True.
    n_jobs : int | 1
        Number of jobs to use for parallel computing (use -1 to use all
        jobs). The parallel loop is set at the pair level.
    kw_links : dict | {}
        Additional arguments for selecting links to compute are passed to the
        function :func:`frites.conn.conn_links`

    Returns
    -------
    te : array_like
        The TE array of shape (n_pairs, max_delay, time - max_delay) if
        return_delays is True or (n_pairs, time - max_delay) if False.

    See also
    --------
    conn_links
    """
    set_log_level(verbose)
    # inputs conversion
    kw_links.update({'directed': True, 'net': False})
    data, cfg = conn_io(
        data, times=times, roi=roi, agg_ch=False, win_sample=None,
        name='Transfert Entropy', verbose=verbose, sfreq=sfreq,
        kw_links=kw_links
    )

    # extract variables
    x, roi, times = data.data, data['roi'].data, data['times'].data
    n_epochs, n_roi, n_pts = data.shape
    x_s, x_t, roi_p, attrs = cfg['x_s'], cfg['x_t'], cfg['roi_p'], cfg['attrs']

    # transpose x to avoid shape checking (n_roi, n_times, 1, n_trials)
    x = x.transpose(1, 2, 0)[..., np.newaxis, :]

    # define delay range
    delays = np.arange(-max_delay, -min_delay, step_delay)[::-1]

    # apply copnorm across trials
    if gcrn:
        x = copnorm_nd(x, axis=-1)

    # build parallel function
    parallel, p_fun = parallel_func(_para_te, n_jobs=n_jobs, verbose=verbose,
                                    total=len(x_s), mesg='Estimating TE')

    # compute the transfer entropy
    logger.info(f"Compute Transfer Entropy (n_pairs={len(x_s)}, "
                f"delays=[{min_delay}:{step_delay}:{max_delay}])")
    te = parallel(
        p_fun(x[n_s, ...], x[n_t, ...], max_delay, return_delays,
              delays) for n_s, n_t in zip(x_s, x_t))
    te = np.stack(te)

    # build coordinates and attributes
    cdelays = np.arange(min_delay, max_delay, step_delay).astype(int) + 1
    ctimes = times[max_delay::]
    _attrs = {
        'max_delay': max_delay, 'min_delay': min_delay,
        'step_delay': step_delay, 'type': 'TE', 'gcrn': gcrn
    }
    attrs = check_attrs({**_attrs, **attrs})

    # mean (or not) over delays
    if return_delays:
        dims, coords = ('roi', 'delays', 'times'), (roi_p, cdelays, ctimes)
    else:
        dims, coords = ('roi', 'times'), (roi_p, ctimes)

    # xarray conversion
    te = xr.DataArray(te, dims=dims, coords=coords, attrs=attrs,
                      name='Transfert Entropy')

    return te


if __name__ == '__main__':
    from frites.simulations import StimSpecAR
    from frites import set_mpl_style

    import matplotlib.pyplot as plt

    set_mpl_style()

    ss = StimSpecAR()
    ar = ss.fit(ar_type='hga', n_epochs=500, n_stim=2, n_std=3)

    te = conn_te(ar, times='times', roi='roi', max_delay=100,
                 return_delays=True, step_delay=3, min_delay=3)

    plt.figure(figsize=(28, 6))
    plt.subplot(141)
    ar.mean('trials').plot(x='times', hue='roi')
    plt.subplot(142)
    te.mean('delays').plot(x='times', hue='roi')
    plt.subplot(143)
    te.sel(roi='x->y').plot()
    plt.subplot(144)
    te.sel(roi='y->x').plot()
    plt.tight_layout()
    plt.show()
