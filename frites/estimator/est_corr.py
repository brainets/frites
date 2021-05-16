"""Correlation based estimators."""
import numpy as np

from frites.estimator.est_mi_base import BaseMIEstimator
from frites.utils import jit


class CorrEstimator(BaseMIEstimator):

    """Correlation-based estimator.

    This estimator can be used to estimate the correlation between two
    continuous variables (mi_type='cc').
    """

    def __init__(self, verbose=None):
        """Init."""
        self.name = 'Correlation-based Estimator'
        super(CorrEstimator, self).__init__(mi_type='cc', verbose=verbose)
        self._core_fun = correlate
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

        This can be particulary usefull when computing correlation in parallel
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

            # types checking
            if x.dtype != np.float32:
                x = x.astype(np.float32, copy=False)
            if y.dtype != np.float32:
                y = y.astype(np.float32, copy=False)
            if categories.dtype != np.int32:
                categories = categories.astype(np.int32, copy=False)

            return core_fun(x, y, categories)

        return estimator


@jit("f4[:, :](f4[:,:,:], f4[:,:,:], i4[:])")
def correlate(x, y, categories):
    """3D correlation."""
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
