"""Transfer entropy using the Gaussian-Copula."""
import numpy as np

from frites.core import cmi_nd_ggg, copnorm_nd
from frites.config import CONFIG


def conn_transfer_entropy(x, max_delay=30, pairs=None, gcrn=True):
    """Across-trials transfer entropy.

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
    assert isinstance(max_delay, (int, int)), ("`max_delay` should be an "
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
