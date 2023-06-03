"""Connectivity using Interaction Information"""
import numpy as np
import xarray as xr

from frites.conn import conn_io, _conn_mi
from frites.core import copnorm_nd
from frites.io import set_log_level, logger, check_attrs
from frites.config import CONFIG

from mne.utils import ProgressBar


def conn_ii(data, y, roi=None, times=None, mi_type='cc', gcrn=True, dt=1,
            verbose=None, **kw_links):
    """Interaction Information on connectivity pairs and behavioral variable.

    This function can be used to investigate if pairs of brain regions (or
    recordings) are mainly carrying the same information, i.e. redundant
    information about a variable of the task (e.g. stimulus, outcome,
    behavioral models) or complementary information, i.e. synergistic. The II
    is defined by :

    .. math::

        II =  I([R_{1}, R_{2}]; S) - I(R_{1}; S) - I(R_{2}; S)

    With :math:`R_{1}, R_{2} and S` the brain activity of two regions and the
    task-related variable.

    .. note::
        **Positive values of II reflects a prevalence of synergistic
        interactions while negative values of II reflects a prevalence of
        redundant interactions.**

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
    kw_links : dict | {}
        Additional arguments for selecting links to compute are passed to the
        function :func:`frites.conn.conn_links`

    Returns
    -------
    interinfo : array_like
        The interaction information in the network of shape (n_pairs, n_times)

    References
    ----------
    McGill 1954 :cite:`mcgill1954`

    See also
    --------
    conn_links, conn_pid
    """
    # _________________________________ INPUTS ________________________________
    # inputs conversion
    kw_links.update({'directed': False, 'net': False})
    data, cfg = conn_io(
        data, y=y, times=times, roi=roi, agg_ch=False, win_sample=None,
        name='II', verbose=verbose, kw_links=kw_links
    )

    # extract variables
    x, attrs = data.data, cfg['attrs']
    y, roi, times = data['y'].data, data['roi'].data, data['times'].data
    x_s, x_t = cfg['x_s'], cfg['x_t']
    roi_p, n_pairs = cfg['roi_p'], len(x_s)

    # build the indices when using multi-variate mi
    assert dt >= 1
    idx = np.mgrid[0:len(times) - dt + 1, 0:dt].sum(0)
    times = times[idx].mean(1)
    n_trials, n_roi, n_times = len(y), len(roi), len(times)

    # copnorm the data
    if gcrn:
        logger.info("    Apply the Gaussian Copula Rank Normalization")
        x = copnorm_nd(x, axis=0)
        if mi_type == 'cc':
            y = copnorm_nd(y, axis=0)

    # reshape y variable
    # if (mi_type == 'cc') and (y.ndim in (1, 2)):
    #     y = np.atleast_2d(y)[np.newaxis, ...]
    #     y = np.tile(y, (x.shape[0], 1, 1))

    # transpose the data to be (n_roi, n_times, 1, n_trials)
    x = np.transpose(x, (1, 2, 0))

    logger.info(f"Compute II on {n_pairs} connectivity pairs")

    # __________________________________ II __________________________________
    # optional argument of gcmi
    kw_mi = CONFIG['KW_GCMI'].copy()
    kw_mi['minorm'] = False

    # compute mi on each node of the network
    pbar = ProgressBar(range(n_roi + n_pairs),
                       mesg='Estimating MI on each node I(X;S)')

    mi_node = np.zeros((n_roi, n_times), dtype=float)
    for n_r in range(n_roi):
        mi_node[n_r, :] = _conn_mi(x[n_r, idx, :], y, mi_type, **kw_mi)
        pbar.update_with_increment_value(1)

    pbar._tqdm.desc = 'Estimating total information I(X,Y;S)'
    infotot = np.zeros((n_pairs, n_times))
    for n_p, (s, t) in enumerate(zip(x_s, x_t)):
        _x_s, _x_t = x[s, ...], x[t, ...]

        # total information estimation
        x_st = np.concatenate((_x_s[idx, ...], _x_t[idx, ...]), axis=1)
        infotot[n_p, :] = _conn_mi(x_st, y, mi_type, **kw_mi)

        pbar.update_with_increment_value(1)

    # interaction information
    interinfo = infotot - mi_node[x_s, :] - mi_node[x_t, :]

    # _______________________________ OUTPUTS _________________________________
    kw = dict(dims=('roi', 'times'), coords=(roi, times),
              attrs=check_attrs(attrs))
    kw_pairs = dict(dims=('roi', 'times'), coords=(roi_p, times))
    interinfo = xr.DataArray(interinfo, name='II',
                             **kw_pairs)
    interinfo.attrs['unit'] = 'bits'

    return interinfo


if __name__ == '__main__':
    from frites.simulations import StimSpecAR
    import matplotlib.pyplot as plt

    ar_type = 'hga'
    n_stim = 2
    n_epochs = 400

    # Simulate an AR model stimulus-specific HGA with redundancy
    # larger than synergy, that is negative Interaction Information
    ss = StimSpecAR()
    ar = ss.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim)

    plt.figure(figsize=(12, 6))

    interinfo = conn_ii(
        ar, 'trials', roi='roi', times='times', mi_type='cd', dt=1,
        verbose=True)

    interinfo.plot()

    plt.show()
