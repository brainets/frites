"""GCMI estimator."""
import numpy as np

from frites.estimator.est_mi_base import BaseMIEstimator
from frites.core.copnorm import copnorm_cat_nd
from frites.core.gcmi_nd import (mi_nd_gg, mi_model_nd_gd, cmi_nd_ggd,
                                 cmi_nd_ggg)
from frites.core.gcmi_1d import (mi_1d_gg, mi_model_1d_gd, cmi_1d_ggd,
                                 cmi_1d_ggg)
from frites.utils import nonsorted_unique


class GCMIEstimator(BaseMIEstimator):

    """Gaussian Copula Mutual-Information estimator.

    Parameters
    ----------
    mi_type : {'cc', 'cd', 'ccd', 'ccc'}
        Mutual information type (default : 'cc') :

            * 'cc' : MI between two continuous variables
            * 'cd' : MI between a continuous and a discret variables
            * 'ccd' : MI between two continuous variables conditioned by a
              third discret one
            * 'ccc' : MI between two continuous variables conditioned by a
              third continuous one

    copnorm : bool | True
        Apply the gaussian-copula rank normalization (default : True)
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
        (default : True)
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized) (default : False)
    tensor : bool | True
        Specify the implementation of the GCMI, either tensor-based (Nd) or
        vector-based (1d). Usually, tensor-based implementation is faster but
        also requires more RAM (default : True)
    gpu : bool | False
        Specify whether the mutual-information has to be computed on CPU
        (gpu=False) or on GPU (gpu=True) (default : False)
    """

    def __init__(self, mi_type='cc', copnorm=True, biascorrect=True,
                 demeaned=False, tensor=True, gpu=False, verbose=None):
        self.name = 'Gaussian Copula Mutual Information Estimator'
        add_str = (f", copnorm={copnorm}, biascorrect={biascorrect}, "
                   f"demeaned={demeaned}")
        super(GCMIEstimator, self).__init__(
            mi_type=mi_type, verbose=verbose, add_str=add_str)

        # =========================== Core function ===========================

        # get the core function for computing MI
        if gpu:
            raise NotImplementedError()
        else:
            if tensor:
                fcn = {
                    'cc': mi_nd_gg, 'cd': mi_model_nd_gd, 'ccd': cmi_nd_ggd,
                    'ccc': cmi_nd_ggg}[mi_type]
            else:
                fcn = {
                    'cc': mi_gg_loop, 'cd': mi_gd_loop, 'ccd': mi_ggd_loop,
                    'ccc': mi_ggg_loop}[mi_type]
        self._core_fun = fcn

        # ========================== Function kwargs ==========================

        # additional arguments that are going to be passed to the core function
        if tensor:
            self._kwargs = dict(biascorrect=biascorrect, demeaned=demeaned,
                                shape_checking=False, traxis=-1, mvaxis=-2)
        else:
            self._kwargs = dict(biascorrect=biascorrect, demeaned=demeaned)
        # update internal settings
        settings = dict(tensor=tensor, gpu=gpu, mi_type=mi_type,
                        copnorm=copnorm, core_fun=self._core_fun.__name__)
        self.settings.merge([self._kwargs, settings])

    def estimate(self, x, y, z=None, categories=None):
        """Estimate the (possibly conditional) mutual-information.

        This method is made for computing the mutual-information on 3D
        variables (i.e (n_var, n_mv, n_samples)) where n_var is an additional
        dimension (e.g times, times x freqs etc.), n_mv is a multivariate
        axis and n_samples the number of samples. When computing MI, both the
        multivariate and samples axes are reduced.

        Parameters
        ----------
        x : array_like
            Array of shape (n_var, n_mv, n_samples). If x has more than three
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
        kwargs = self._kwargs
        core_fun = self._core_fun
        copnorm = self.settings['copnorm']
        mi_type = self.settings['mi_type']

        def estimator(x, y, z=None, categories=None):
            # copnorm the data (if needed)
            if copnorm:
                x = copnorm_cat_nd(x, categories, axis=-1)
                if mi_type == 'cc':
                    y = copnorm_cat_nd(y, categories, axis=-1)
                if (mi_type == 'ccc') and (z is not None):
                    z = copnorm_cat_nd(z, categories, axis=-1)

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

            # repeat y and z(if needed)
            if (mi_type != 'cd') and (y.ndim == 1):
                n_var, n_mv, _ = x.shape
                y = np.tile(y, (n_var, 1, 1))
            if (mi_type == 'ccc') and (y.ndim == 1):
                n_var, n_mv, _ = x.shape
                z = np.tile(z, (n_var, 1, 1))

            # compute (potentially categorical) MI
            n_var = x.shape[0]
            args = ()
            if isinstance(categories, np.ndarray):
                # get unique non-sorted categories
                u_cat = nonsorted_unique(categories)
                # compute per category
                mi = np.zeros((len(u_cat), n_var), dtype=x.dtype)
                for n_c, c in enumerate(u_cat):
                    is_cat = categories == c
                    if mi_type in ['ccd', 'ccc']:
                        args = [z[..., is_cat]]
                    mi[n_c, :] = core_fun(x[..., is_cat], y[..., is_cat],
                                          *args, **kwargs)
            else:
                if mi_type in ['ccd', 'ccc']:
                    args = [z]
                mi = core_fun(x, y, *args, **kwargs)[np.newaxis, :]

            # retrieve original shape (if needed)
            if reshape is not None:
                mi = mi.reshape([mi.shape[0]] + reshape[0])

            return mi.astype(np.float32)
        return estimator


def mi_gg_loop(x, y, **kw):
    """I(C; C) 1d loop."""
    n_var = x.shape[0]
    mi = np.zeros((n_var), dtype=x.dtype)
    for k in range(n_var):
        mi[k] = mi_1d_gg(x[k, ...], y[k, ...], **kw)
    return mi


def mi_gd_loop(x, y, **kw):
    """I(C; D) 1d loop."""
    n_var = x.shape[0]
    mi = np.zeros((n_var), dtype=x.dtype)
    for k in range(n_var):
        mi[k] = mi_model_1d_gd(x[k, ...], y, **kw)
    return mi


def mi_ggd_loop(x, y, z, **kw):
    """I(C; C | D) 1d loop."""
    n_var = x.shape[0]
    mi = np.zeros((n_var), dtype=x.dtype)
    for k in range(n_var):
        mi[k] = cmi_1d_ggd(x[k, ...], y[k, ...], z, **kw)
    return mi


def mi_ggg_loop(x, y, z, **kw):
    """I(C; C | C) 1d loop."""
    n_var = x.shape[0]
    mi = np.zeros((n_var), dtype=x.dtype)
    for k in range(n_var):
        mi[k] = cmi_1d_ggg(x[k, ...], y[k, ...], z[k, ...], **kw)
    return mi
