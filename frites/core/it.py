"""Information transfer functions."""
import numpy as np
import xarray as xr

from frites.io import set_log_level, logger

from frites.core import mi_nd_gg, cmi_nd_ggg, copnorm_nd
from frites.utils import jit
from frites.config import CONFIG



###############################################################################
###############################################################################
#                     DYNAMIC FUNCTIONAL CONNECTIVITY
###############################################################################
###############################################################################


def dfc_gc(data, times, roi, win_sample, verbose=None):
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


###############################################################################
###############################################################################
#                              TRANFSER ENTROPY
###############################################################################
###############################################################################


def it_transfer_entropy(x, max_delay=30, pairs=None, gcrn=True):
    """Compute the transfer entropy.

    The transfer entropy represents the amount of information that is send
    from a source to a target. It is defined as :

    .. math::

        TE = I(source_{past}; target_{present} | target_{past})

    Where :math:`past` is defined using the `max_delay` input parameter. Note
    that the transfer entropy only provides about the amount of information
    that is sent, not on the content.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_roi, n_times, n_epochs). Must be a gaussian
        variable
    max_delay : int | 30
        Number of time points defining where to stop looking at in the past.
        Increasing this maximum delay input can lead to slower computations
    pairs : array_like
        Array of pairs to consider for computing the transfer entropy. It
        should be an array of shape (n_pairs, 2) where the first column refers
        to sources and the second to targets. If None, all pairs will be
        computed
    gcrn : bool | True
        Apply a Gaussian Copula rank normalization

    Returns
    -------
    te : array_like
        The transfer entropy array of shape (n_pairs, n_times - max_delay)
    pairs : array_like
        Pairs vector use for computations of shape (n_pairs, 2)
    """
    # -------------------------------------------------------------------------
    # check pairs
    n_roi, n_times, n_epochs = x.shape
    if not isinstance(pairs, np.ndarray):
        pairs = np.c_[np.where(~np.eye(n_roi, dtype=bool))]
    assert isinstance(pairs, np.ndarray) and (pairs.ndim == 2) and (
        pairs.shape[1] == 2), ("`pairs` should be a 2d array of shape "
                               "(n_pairs, 2) where the first column refers to "
                               "sources and the second to targets")
    x_all_s, x_all_t = pairs[:, 0], pairs[:, 1]
    n_pairs = len(x_all_s)
    # check max_delay
    assert isinstance(max_delay, (int, np.int)), ("`max_delay` should be an "
                                                  "integer")
    # check input data
    assert (x.ndim == 3), ("input data `x` should be a 3d array of shape "
                           "(n_roi, n_times, n_epochs)")
    x = x[..., np.newaxis, :]

    # -------------------------------------------------------------------------
    # apply copnorm
    if gcrn:
        x = copnorm_nd(x, axis=-1)

    # -------------------------------------------------------------------------
    # compute the transfer entropy
    te = np.zeros((n_pairs, n_times - max_delay), dtype=float)
    for n_s, x_s in enumerate(x_all_s):
        # select targets
        is_source = x_all_s == x_s
        x_t = x_all_t[is_source]
        targets = x[x_t, ...]
        # tile source
        source = np.tile(x[[x_s], ...], (targets.shape[0], 1, 1, 1))
        # loop over remaining time points
        for n_d, d in enumerate(range(max_delay + 1, n_times)):
            t_pres = np.tile(targets[:, [d], :], (1, max_delay, 1, 1))
            past = slice(d - max_delay - 1, d - 1)
            s_past = source[:, past, ...]
            t_past = targets[:, past, ...]
            # compute the transfer entropy
            _te = cmi_nd_ggg(s_past, t_pres, t_past, **CONFIG["KW_GCMI"])
            # take the sum over delays
            te[is_source, n_d] = _te.mean(1)

    return te, pairs


###############################################################################
###############################################################################
#                 FEATURE SPECIFIC INFORMATION TRANSFER
###############################################################################
###############################################################################


@jit("f4[:,:,:](f4[:,:,:], f4[:,:,:], f4[:], f4)")
def it_fit(x_s, x_t, times, max_delay):  # noqa
    """Compute Feature-specific Information Transfer (FIT).

    This function has been written for supporting 3D arrays. If Numba is
    installed, performances of this function can be greatly improved.

    Parameters
    ----------
    x_s : array_like
        Array to use as source. Must be a 3d array of shape (:, :, n_times)
        and of type np.float32
    x_t : array_like
        Array to use as target. Must be a 3d array of shape (:, :, n_times)
        and of type np.float32
    times : array_like
        Time vector of shape (n_times,) and of type np.float32
    max_delay : float | .3
        Maximum delay (must be a np.float32)

    Returns
    -------
    fit : array_like
        Array of FIT of shape (:, :, n_times - max_delay)
    """
    # ---------------------------------------------------------------------
    n_dim, n_suj, n_times = x_s.shape
    # time indices for target roi
    t_start = np.where(times > times[0] + max_delay)[0]
    # max delay index
    max_delay = n_times - len(t_start)

    # ---------------------------------------------------------------------
    # Compute FIT on original MI values
    fit = np.zeros((n_dim, n_suj, n_times - max_delay), dtype=np.float32)

    # mi at target roi in the present
    x_t_pres = x_t[:, :, t_start]

    # Loop over delays for past of target and sources
    for delay in range(1, max_delay):
        # get past delay indices
        past_delay = t_start - delay
        # mi at target roi in the past
        x_t_past = x_t[:, :, past_delay]
        # mi at sources roi in the past
        x_s_past = x_s[:, :, past_delay]
        # redundancy between sources and target (min MI)
        red_s_t = np.minimum(x_t_pres, x_s_past)
        # redundancy between sources, target present and target past
        red_all = np.minimum(red_s_t, x_t_past)
        # sum delay-specific FIT (source, target)
        fit += red_s_t - red_all

    return fit
