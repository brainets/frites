"""Mutual information using binnings.

All the functions inside this file can be compiled using Numba.
"""
import numpy as np
import logging

from frites.utils import jit

logger = logging.getLogger('frites')


###############################################################################
###############################################################################
#                        LOW-LEVEL CORE FUNCTIONS
###############################################################################
###############################################################################
"""
This first part contains functions for computing the mutual information using
binning method. In particular it redefines sub-functions that can be compiled
using Numba (e.g histogram 1D and 2D).
"""


@jit("f4(f4[:])")
def entropy(x):
    """Compute the entropy of a continuous variable.

    Parameters
    ----------
    x : array_like
        Distribution of probabilities of shape (N,) and of type np.float32

    Returns
    -------
    entr : np.float32
        Entropy of the distribution
    """
    x_max, x_min = x.max(), x.min()
    assert (x_min >= 0) and (x_max <= 1)
    if x_min == x_max == 0:
        return np.float32(0.)
    # Take only non-zero values as log(0) = 0 :
    nnz_x = x[np.nonzero(x)]
    entr = -np.sum(nnz_x * np.log2(nnz_x))

    return entr


@jit("i8[:](f4[:], i8)")
def histogram(x, bins):
    """Compute the histogram of a continuous row vector.

    Parameters
    ----------
    x : array_like
        Vector array of shape (N,) and of type np.float32
    bins : int64
        Number of bins

    Returns
    -------
    hist : array_like
        Vector array of shape (bins,) and of type int64
    """
    hist = np.histogram(x, bins=bins)[0]
    return hist


@jit("i8[:,:](f4[:], f4[:], i8, i8)")
def histogram2d(x, y, bins_x, bins_y):
    """Histogram 2d between two continuous row vectors.

    Parameters
    ----------
    x : array_like
        Vector array of shape (N,) and of type np.float32
    y : array_like
        Vector array of shape (N,) and of type np.float32
    bins_x, bins_y : int64
        Number of bins respectively for the x and y variables

    Returns
    -------
    hist : array_like
        Array of shape (bins, bins) and of type int64
    """
    # x-range
    x_max, x_min = x.max(), x.min()
    delta_x = 1 / ((x_max - x_min) / bins_x)
    # y-range
    y_max, y_min = y.max(), y.min()
    delta_y = 1 / ((y_max - y_min) / bins_y)
    # compute histogram 2d
    xy_bin = np.zeros((np.int64(bins_x), np.int64(bins_y)), dtype=np.int64)
    for t in range(len(x)):
        i = (x[t] - x_min) * delta_x
        j = (y[t] - y_min) * delta_y
        if 0 <= i < bins_x and 0 <= j < bins_y:
            xy_bin[int(i), int(j)] += 1
    return xy_bin


@jit("f4(f4[:], f4[:], i8, i8)")
def mi_bin(x, y, bins_x, bins_y):
    """Mutual information between two arrays I(X; Y) using binning.

    Parameters
    ----------
    x : array_like
        Vector array of shape (N,) and of type np.float32
    y : array_like
        Vector array of shape (N,) and of type np.float32
    bins_x, bins_y : int64
        Number of bins respectively for the x and y variables

    Returns
    -------
    i : np.float32
        The mutual information of type float32
    """
    if bins_y == 0:
        bins_y = len(np.unique(y))
    # compute probabilities
    p_x = histogram(x, bins_x)
    p_y = histogram(y, bins_y)
    p_xy = histogram2d(x, y, bins_x, bins_y)
    p_x = p_x / p_x.sum()
    p_y = p_y / p_y.sum()
    p_xy = p_xy / p_xy.sum()
    # compute entropy
    h_x = entropy(p_x.astype(np.float32))
    h_y = entropy(p_y.astype(np.float32))
    h_xy = entropy(p_xy.ravel().astype(np.float32))
    # compute mutual information
    i = h_x + h_y - h_xy

    return i


@jit("f4(f4[:], f4[:], f4[:], i8)")
def mi_bin_ccd(x, y, z, bins):
    """Compute the conditional mutual information I(X; Y | Z) using binning.

    Parameters
    ----------
    x, y, z : array_like
        Vector arrays of shape (N,) and of type np.float32
    bins : int64
        Number of bins

    Returns
    -------
    cmi : np.float32
        The conditional mutual information of type float32
    """
    # get unique z elements
    z_u = np.unique(z)
    n_z = len(z_u)
    # compute mi for each elements of z
    pz = np.zeros((np.int64(n_z)), dtype=np.float32)
    icond = np.zeros((np.int64(n_z)), dtype=np.float32)
    for n_k, k in enumerate(z_u):
        idx_z = z == k
        pz[n_k] = idx_z.sum()
        _x, _y = x[idx_z], y[idx_z]
        icond[n_k] = mi_bin(_x, _y, bins, bins)
    # conditional mutual information
    pz /= len(z)
    cmi = np.sum(pz * icond)

    return cmi


###############################################################################
###############################################################################
#                         MID-LEVEL CORE FUNCTIONS
###############################################################################
###############################################################################
"""
This second part defines mid level core mi functions mainly to speed up
computations on arrays that have a time dimension.
"""


@jit("f4[:](f4[:,:], f4[:], i8, i8)")
def mi_bin_time(x, y, bins_x, bins_y):
    """Compute the MI between two variables across time.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_times, n_trials)
    y : array_like
        Regressor array of shape (n_trials)
    bins_x, bins_y : int64
        Number of bins respectively for the x and y variables

    Returns
    -------
    mi : array_like
        Array of mutual information of shape (n_times)
    """
    n_times, n_trials = x.shape
    mi = np.zeros((n_times), dtype=np.float32)
    for t in range(n_times):
        mi[t] = mi_bin(x[t, :], y, bins_x, bins_y)
    return mi


@jit("f4[:](f4[:,:], f4[:], f4[:], i8)")
def mi_bin_ccd_time(x, y, z, bins):
    """Compute the MI between two variables across time.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_times, n_trials)
    y : array_like
        Regressor array of shape (n_trials)
    z : array_like
        Conditional array of shape (n_trials)
    bins : int64
        Number of bins

    Returns
    -------
    cmi : array_like
        Array of conditional mutual information of shape (n_times)
    """
    n_times, n_trials = x.shape
    mi = np.zeros((n_times), dtype=np.float32)
    for t in range(n_times):
        mi[t] = mi_bin_ccd(x[t, :], y, z, bins)
    return mi


@jit("f4[:](f4[:,:], f4[:,:], i8, i8)")
def mi_bin_conn_time(x, y, bins_x, bins_y):
    """Compute the MI between two variables of equal shapes across time.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_times, n_trials)
    y : array_like
        Regressor array of shape (n_times, n_trials)
    bins_x, bins_y : int64
        Number of bins respectively for the x and y variables

    Returns
    -------
    mi : array_like
        Array of mutual information of shape (n_times)
    """
    n_times, n_trials = x.shape
    mi = np.zeros((n_times), dtype=np.float32)
    for t in range(n_times):
        mi[t] = mi_bin(x[t, :], y[t, :], bins_x, bins_y)
    return mi
