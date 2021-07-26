"""Gaussian-copula rank normalization functions."""
import numpy as np
from scipy.special import ndtri

from frites.config import CONFIG


def ctransform(x):
    """Copula transformation (empirical CDF).

    Parameters
    ----------
    x : array_like
        Array of data. The trial axis should be the last one

    Returns
    -------
    xr : array_like
        Empirical CDF value along the last axis of x. Data is ranked and scaled
        within [0 1] (open interval)
    """
    xr = np.argsort(np.argsort(x)).astype(float)
    xr += 1.
    xr /= float(xr.shape[-1] + 1)
    return xr


def copnorm_1d(x):
    """Copula normalization for a single vector.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_epochs,)

    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
    """
    assert isinstance(x, np.ndarray) and (x.ndim == 1)
    return ndtri(ctransform(x))


def copnorm_cat_1d(x, y):
    """Categorical Copula normalization for a single vector.

    This function apply the copnorm per categories.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_epochs,)
    y : array_like
        Array of shape (n_epochs,) of integers describing the categories.

    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
    """
    assert isinstance(x, np.ndarray) and (x.ndim == 1)
    assert isinstance(y, np.ndarray) and (x.ndim == 1)
    assert y.dtype in CONFIG['INT_DTYPE']
    x_cop = np.zeros_like(x)
    y_u = np.unique(y)
    for yi in y_u:
        _idx = y == yi
        x_cop[_idx] = copnorm_1d(x[_idx])
    return x_cop


def copnorm_nd(x, axis=-1):
    """Copula normalization for a multidimentional array.

    Parameters
    ----------
    x : array_like
        Array of data
    axis : int | -1
        Epoch (or trial) axis. By default, the last axis is considered

    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
    """
    assert isinstance(x, np.ndarray) and (x.ndim >= 1)
    return np.apply_along_axis(copnorm_1d, axis, x)


def copnorm_cat_nd(x, y, axis=-1):
    """Categorical Copula normalization for multidimentional array.

    This function apply the copnorm per categories.

    Parameters
    ----------
    x : array_like
        Array of data
    y : array_like
        Array of shape (n_epochs,) of integers describing the categories.
    axis : int | -1
        Epoch (or trial) axis. By default, the last axis is considered

    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
    """
    assert isinstance(x, np.ndarray) and (x.ndim >= 1)
    if y is None:
        return copnorm_nd(x, axis=axis)
    return np.apply_along_axis(copnorm_cat_1d, axis, x, y)
