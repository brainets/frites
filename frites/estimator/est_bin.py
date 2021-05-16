"""Binning-based MI estimator."""
import numpy as np

from frites.core.mi_bin_ephy import (mi_bin_time, mi_bin_ccd_time)
from frites.estimator.est_mi_base import BaseMIEstimator
from frites.utils import jit


class BinMIEstimator(BaseMIEstimator):

    """Binning-based Mutual-Information estimator.

    .. note::

        The functions for estimating the mutual-information using binning are
        relatively slow. If Numba is installed, those functions can be
        considerably accelerated.

    Parameters
    ----------
    mi_type : {'cc', 'cd', 'ccd'}
        Mutual information type (default : 'cc') :

            * 'cc' : MI between two continuous variables
            * 'cd' : MI between a continuous and a discret variables
            * 'ccd' : MI between two continuous variables conditioned by a
              third discret one

    n_bins : int | 4
        Number of bins to estimate the probability distribution.
    """

    def __init__(self, mi_type='cc', n_bins=4, verbose=None):
        self.name = 'Binning-based Mutual Information Estimator'
        add_str = f", n_bins={n_bins}"
        super(BinMIEstimator, self).__init__(
            mi_type=mi_type, add_str=add_str, verbose=verbose)

        # =========================== Core function ===========================
        fcn = {'cc': mi_bin_cc, 'cd': mi_bin_cd, 'ccd': mi_bin_ccd}[mi_type]
        self._core_fun = fcn

        # ========================== Function kwargs ==========================

        # additional arguments that are going to be passed to the core function
        self._kwargs = dict(n_bins=n_bins)
        # update internal settings
        settings = dict(mi_type=mi_type, core_fun=self._core_fun.__name__)
        self.settings.merge([self._kwargs, settings])

    def estimate(self, x, y, z=None, categories=None):
        """Estimate the (possibly conditional) mutual-information.

        This method is made for computing the mutual-information on 3D
        variables (i.e (n_var, 1, n_samples)) where n_var is an additional
        dimension (e.g times, times x freqs etc.), 1 is a multivariate
        axis and n_samples the number of samples. When computing MI, both the
        multivariate and samples axes are reduced.

        Parameters
        ----------
        x : array_like
            Array of shape (n_var, 1, n_samples). If x has more than three
            dimensions, it's going to be internally reshaped.
        y : array_like
            Array with a shape that depends on the type of MI (mi_type) :

                * If mi_type is 'cc' or 'ccd', y should be an array with the
                  same shape as x
                * If mi_type is 'cd', y should be a row vector of shape
                  (n_samples,)

        z : array_like | None
            Array for conditional mutual-information. The shape is going to
            depend on the type of MI (mi_type) :

                * If mi_type is 'ccd', z should be a row vector of shape
                  (n_samples,)
                * If mi_type is 'ccc', z should have the same shape as x and y

        categories : array_like | None
            Row vector of categories. This vector should have a shape of
            (n_samples,) and should contains integers describing the category
            of each sample. If categories are provided, the copnorm is going to
            be performed per categories.

        Returns
        -------
        mi : array_like
            Array of (possibly conditional) mutual-information of shape
            (n_categories, n_var). If categories is None when computing MI,
            n_categories is going to be one.
        """
        fcn = self.get_function()
        return fcn(x, y, z=z, categories=categories)

    def get_function(self):
        """Get the function to execute according to the input parameters.

        This can be particulary usefull when computing MI in parallel as it
        avoids to pickle the whole estimator and therefore, leading to faster
        computations.

        The returned function has the following signature :

            * fcn(x, y, z=None, categories=None)

        and return an array of shape (n_categories, n_var).
        """
        n_bins = np.int64(self._kwargs['n_bins'])
        core_fun = self._core_fun
        mi_type = self.settings['mi_type']

        def estimator(x, y, z=None, categories=None):
            # be sure that x is at least 3d
            if x.ndim == 1:
                x = x[np.newaxis, np.newaxis, :]
            if x.ndim == 2:
                x = x[np.newaxis, :]

            # internal reshaping if x has more than 3 dimensions
            assert x.ndim >= 3
            reshape = None
            if x.ndim > 3:
                head_shape = list(x.shape)[0:-2]
                reshape = (head_shape, np.prod(head_shape))
                tail_shape = list(x.shape)[-2::]
                x = x.reshape([reshape[1]] + tail_shape)

            # types checking
            if x.dtype != np.float32:
                x = x.astype(np.float32, copy=False)
            if y.dtype != np.float32:
                y = y.astype(np.float32, copy=False)
            if isinstance(z, np.ndarray) and (z.dtype != np.float32):
                z = z.astype(np.float32, copy=False)
            if not isinstance(categories, np.ndarray):
                categories = np.zeros((1), dtype=np.float32)
            if categories.dtype != np.float32:
                categories = categories.astype(np.float32, copy=False)

            # additional arguments for cmi
            args = ()
            if mi_type in ['ccd', 'ccc']:
                args = [z]

            # compute mi
            mi = core_fun(x, y, *args, n_bins, categories)

            # retrieve original shape (if needed)
            if reshape is not None:
                mi = mi.reshape([mi.shape[0]] + reshape[0])

            return mi
        return estimator


@jit("f4[:, :](f4[:,:,:], f4[:], i8, f4[:])")
def mi_bin_cc(x, y, n_bins, categories):
    # proper shape of the regressor
    n_times, _, n_trials = x.shape
    # compute mi across (ffx) or per subject (rfx)
    if len(categories) != n_trials:
        mi = np.zeros((1, n_times), dtype=np.float32)
        mi[0, :] = mi_bin_time(x[:, 0, :], y, n_bins, n_bins)
    else:
        # get categories informations
        u_cat = np.unique(categories)
        n_cats = len(u_cat)
        # compute mi per subject
        mi = np.zeros((n_cats, n_times), dtype=np.float32)
        for n_c, c in enumerate(u_cat):
            is_cat = categories == c
            x_cat, y_cat = x[:, :, is_cat], y[is_cat]
            mi[n_c, :] = mi_bin_time(x_cat[:, 0, :], y_cat, n_bins, n_bins)
    return mi


@jit("f4[:, :](f4[:,:,:], f4[:], i8, f4[:])")
def mi_bin_cd(x, y, bins_x, categories):
    # proper shape of the regressor
    n_times, _, n_trials = x.shape
    # get the number of bins for the y variable
    bins_y = len(np.unique(y))
    # compute mi across (ffx) or per subject (rfx)
    if len(categories) != n_trials:
        mi = np.zeros((1, n_times), dtype=np.float32)
        mi[0, :] = mi_bin_time(x[:, 0, :], y, bins_x, bins_y)
    else:
        # get categories informations
        u_cat = np.unique(categories)
        n_cats = len(u_cat)
        # compute mi per subject
        mi = np.zeros((n_cats, n_times), dtype=np.float32)
        for n_c, c in enumerate(u_cat):
            is_cat = categories == c
            x_cat, y_cat = x[:, :, is_cat], y[is_cat]
            mi[n_c, :] = mi_bin_time(x_cat[:, 0, :], y_cat, bins_x, bins_y)
    return mi


@jit("f4[:, :](f4[:,:,:], f4[:], f4[:], i8, f4[:])")
def mi_bin_ccd(x, y, z, n_bins, categories):
    # proper shape of the regressor
    n_times, _, n_trials = x.shape
    # compute mi across (ffx) or per subject (rfx)
    if len(categories) != n_trials:
        mi = np.zeros((1, n_times), dtype=np.float32)
        mi[0, :] = mi_bin_ccd_time(x[:, 0, :], y, z, n_bins)
    else:
        # get categories informations
        u_cat = np.unique(categories)
        n_cats = len(u_cat)
        # compute mi per subject
        mi = np.zeros((n_cats, n_times), dtype=np.float32)
        for n_c, c in enumerate(u_cat):
            is_cat = categories == c
            x_cat, y_cat, z_cat = x[:, :, is_cat], y[is_cat], z[is_cat]
            mi[n_c, :] = mi_bin_ccd_time(x_cat[:, 0, :], y_cat, z_cat, n_bins)
    return mi
