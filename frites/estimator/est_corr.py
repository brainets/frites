"""Correlation based estimators."""
import numpy as np
from scipy.stats import spearmanr

from frites.estimator.est_mi_base import BaseMIEstimator
# from frites.utils import jit


class CorrEstimator(BaseMIEstimator):

    """Correlation-based estimator.

    This estimator can be used to estimate the correlation between two
    continuous variables (mi_type='cc').

    Parameters
    ----------
    method : {'pearson', 'spearman'}
        Use either the Pearson correlation or the rank-based Spearman
        correlation
    implementation : {'vector', 'tensor'}
        Specify whether to use the traditional vector-based implementation or
        the tensor-based implementation (usually faster)
    """

    def __init__(self, method='pearson', implementation='vector',
                 verbose=None):
        """Init."""
        self.name = 'Correlation-based Estimator'

        # implementation selection
        _methods = {
            'pearson': {
                'tensor': ten_pearson,
                'vector': vec_pearson,
                # 'numba': nb_pearson
            },
            'spearman': {
                'tensor': ten_spearman,
                'vector': vec_spearman,
                # 'numba': vec_spearman
            }
        }
        self._method = method
        self._implementation = implementation
        self._core_fun = _methods[method][implementation]
        # additional string for the description
        add_str = f", method={method}, implementation={implementation}"
        super(CorrEstimator, self).__init__(mi_type='cc', verbose=verbose,
                                            add_str=add_str)
        # update internal settings
        settings = dict(mi_type='cc', core_fun=self._core_fun.__name__)
        self.settings.merge([settings])

    def estimate(self, x, y, z=None, categories=None):
        """Estimate the correlation between two variables.

        This method is made for computing the correlation on 3D variables
        (i.e (n_var, 1, n_samples)) where n_var is an additional dimension
        (e.g times, times x freqs etc.), and n_samples the number of samples.

        Parameters
        ----------
        x, y : array_like
            Array of shape (n_var, 1, n_samples).
        categories : array_like | None
            Row vector of categories. This vector should have a shape of
            (n_samples,) and should contains integers describing the category
            of each sample.

        Returns
        -------
        corr : array_like
            Array of correlation of shape (n_categories, n_var).
        """
        fcn = self.get_function()
        return fcn(x, y, categories=categories)

    def get_function(self):
        """Get the function to execute according to the input parameters.

        This can be particularly useful when computing correlation in parallel
        as it avoids to pickle the whole estimator and therefore, leading to
        faster computations.

        The returned function has the following signature :

            * fcn(x, y, *args, categories=None, **kwargs)

        and return an array of shape (n_categories, n_var).
        """
        core_fun = self._core_fun

        def estimator(x, y, *args, categories=None, **kwargs):
            if categories is None:
                categories = np.array([], dtype=np.float32)

            # be sure that x is at least 3d
            if x.ndim == 1:
                x = x[np.newaxis, np.newaxis, :]
            if x.ndim == 2:
                x = x[np.newaxis, :]

            # repeat y (if needed)
            if (y.ndim == 1):
                n_var, n_mv, _ = x.shape
                y = np.tile(y, (n_var, 1, 1))

            # numba related changes
            if self._implementation == 'numba':
                if x.dtype != np.float32:
                    x = x.astype(np.float32, copy=False)
                if y.dtype != np.float32:
                    y = y.astype(np.float32, copy=False)
                if categories.dtype != np.int32:
                    categories = categories.astype(np.int32, copy=False)

            # change flags (compatibility with multi-core processing)
                x.flags.writeable = True
                y.flags.writeable = True
                categories.flags.writeable = True
                x = np.ascontiguousarray(x)
                y = np.ascontiguousarray(y)
                categories = np.ascontiguousarray(categories)

            return core_fun(x, y, categories)

        return estimator


###############################################################################
###############################################################################
#                                 PEARSON
###############################################################################
###############################################################################


def tpearson(x, y, axis=0):
    """Tensor-based pearson correlation."""
    n = x.shape[axis]
    xc = x - x.mean(axis=axis, keepdims=True)
    yc = y - y.mean(axis=axis, keepdims=True)
    xystd = x.std(axis=axis) * y.std(axis=axis)
    cov = (xc * yc).sum(axis=axis) / n
    corr = cov / xystd
    return corr


def vec_pearson(x, y, categories):
    """Numpy-based pearson correlation."""
    # proper shape of the regressor
    n_times, _, n_trials = x.shape
    if len(categories) != n_trials:
        corr = np.zeros((1, n_times), dtype=np.float32)
        for t in range(n_times):
            corr[0, t] = np.corrcoef(x[t, 0, :], y[t, 0, :])[0, 1]
    else:
        # get categories informations
        u_cat = np.unique(categories)
        n_cats = len(u_cat)
        # compute mi per subject
        corr = np.zeros((n_cats, n_times), dtype=np.float32)
        for n_c, c in enumerate(u_cat):
            is_cat = categories == c
            x_c, y_c = x[:, :, is_cat], y[:, :, is_cat]
            for t in range(n_times):
                corr[n_c, t] = np.corrcoef(x_c[t, 0, :], y_c[t, 0, :])[0, 1]

    return corr


def ten_pearson(x, y, categories):
    """Numpy-based pearson correlation."""
    # proper shape of the regressor
    n_times, _, n_trials = x.shape
    if len(categories) != n_trials:
        corr = tpearson(x, y, axis=2).T
    else:
        # get categories informations
        u_cat = np.unique(categories)
        n_cats = len(u_cat)
        # compute mi per subject
        corr = np.zeros((n_cats, n_times), dtype=np.float32)
        for n_c, c in enumerate(u_cat):
            is_cat = categories == c
            x_c, y_c = x[:, :, is_cat], y[:, :, is_cat]
            corr[n_c, :] = tpearson(x_c, y_c, axis=2).T

    return corr


# @jit("f4[:, :](f4[:,:,:], f4[:,:,:], i4[:])")
# def nb_pearson(x, y, categories):
#     """Numba-based pearson correlation."""
#     # proper shape of the regressor
#     n_times, _, n_trials = x.shape
#     if len(categories) != n_trials:
#         corr = np.zeros((1, n_times), dtype=np.float32)
#         for t in range(n_times):
#             corr[0, t] = np.corrcoef(x[t, 0, :], y[t, 0, :])[0, 1]
#     else:
#         # get categories informations
#         u_cat = np.unique(categories)
#         n_cats = len(u_cat)
#         # compute mi per subject
#         corr = np.zeros((n_cats, n_times), dtype=np.float32)
#         for n_c, c in enumerate(u_cat):
#             is_cat = categories == c
#             x_c, y_c = x[:, :, is_cat], y[:, :, is_cat]
#             for t in range(n_times):
#                 corr[n_c, t] = np.corrcoef(x_c[t, 0, :], y_c[t, 0, :])[0, 1]

#     return corr


###############################################################################
###############################################################################
#                                 SPEARMAN
###############################################################################
###############################################################################


def tspearman(x, y, axis=0):
    """Tensor-based spearman correlation."""
    n = x.shape[axis]
    x = np.argsort(x.argsort(axis=axis))
    y = np.argsort(y.argsort(axis=axis))
    xc = x - x.mean(axis=axis, keepdims=True)
    yc = y - y.mean(axis=axis, keepdims=True)
    xystd = x.std(axis=axis) * y.std(axis=axis)
    cov = (xc * yc).sum(axis=axis) / n
    corr = cov / xystd
    return corr


def vec_spearman(x, y, categories):
    """Numpy-based spearman correlation."""
    # proper shape of the regressor
    n_times, _, n_trials = x.shape
    if len(categories) != n_trials:
        corr = np.zeros((1, n_times), dtype=np.float32)
        for t in range(n_times):
            corr[0, t] = spearmanr(x[t, 0, :], y[t, 0, :])[0]
    else:
        # get categories informations
        u_cat = np.unique(categories)
        n_cats = len(u_cat)
        # compute mi per subject
        corr = np.zeros((n_cats, n_times), dtype=np.float32)
        for n_c, c in enumerate(u_cat):
            is_cat = categories == c
            x_c, y_c = x[:, :, is_cat], y[:, :, is_cat]
            for t in range(n_times):
                corr[n_c, t] = spearmanr(x_c[t, 0, :], y_c[t, 0, :])[0]

    return corr


def ten_spearman(x, y, categories):
    """Numpy-based pearson correlation."""
    # proper shape of the regressor
    n_times, _, n_trials = x.shape
    if len(categories) != n_trials:
        corr = tspearman(x, y, axis=2).T
    else:
        # get categories informations
        u_cat = np.unique(categories)
        n_cats = len(u_cat)
        # compute mi per subject
        corr = np.zeros((n_cats, n_times), dtype=np.float32)
        for n_c, c in enumerate(u_cat):
            is_cat = categories == c
            x_c, y_c = x[:, :, is_cat], y[:, :, is_cat]
            corr[n_c, :] = tspearman(x_c, y_c, axis=2).T

    return corr
