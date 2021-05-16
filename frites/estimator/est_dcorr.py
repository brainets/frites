"""Correlation based estimators."""
import numpy as np

from frites.io import logger
from frites.estimator.est_mi_base import BaseMIEstimator


class DcorrEstimator(BaseMIEstimator):

    """Distance correlation-based estimator.

    This estimator can be used to estimate the correlation between two
    continuous variables (mi_type='cc').

    Parameters
    ----------
    implementation : {'auto', 'frites', 'dcor'}
        Choose wich implementation of the distance correlation to use. If
        'frites' a home-made version is going to be used. If 'dcor', the one of
        the dcorr package is going to be preferred (see for installation
        `<https://dcor.readthedocs.io/>`_).
    """

    def __init__(self, implementation='auto', verbose=None):
        """Init."""
        self.name = 'Distance correlation-based Estimator'
        # get the distance correlation function
        fcn, implementation = get_distance_correlation(
            implementation=implementation)
        self._core_fun = wrap_dcorr(fcn)
        # instantiate base class
        add_str = f", implementation={implementation}"
        super(DcorrEstimator, self).__init__(
            mi_type='cc', add_str=add_str, verbose=verbose)
        # update internal settings
        settings = dict(mi_type='cc', core_fun=self._core_fun.__name__,
                        implementation=implementation)
        self.settings.merge([settings])

    def estimate(self, x, y, z=None, categories=None):
        """Estimate the distance correlation between two variables.

        This method is made for computing the correlation on 3D variables
        (i.e (n_var, n_mv, n_samples)) where n_var is an additional dimension
        (e.g times, times x freqs etc.)n_mv is a multivariate axis and
        n_samples the number of samples.

        Parameters
        ----------
        x, y : array_like
            Array of shape (n_var, n_mv, n_samples).
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

            return core_fun(x, y, categories)

        return estimator


def wrap_dcorr(fcn):
    def correlate(x, y, categories):
        """3D distance correlation."""
        # transpose x and y to be (n_samples, n_mv, n_var)
        x, y = np.transpose(x, (2, 1, 0)), np.transpose(y, (2, 1, 0))
        # proper shape of the regressor
        n_trials, _, n_times = x.shape
        if len(categories) != n_trials:
            corr = np.zeros((1, n_times), dtype=np.float32)
            for t in range(n_times):
                corr[0, t] = fcn(x[:, :, t], y[:, :, t])
        else:
            # get categories informations
            u_cat = np.unique(categories)
            n_cats = len(u_cat)
            # compute mi per subject
            corr = np.zeros((n_cats, n_times), dtype=np.float32)
            for n_c, c in enumerate(u_cat):
                is_cat = categories == c
                x_c, y_c = x[is_cat, :, :], y[is_cat, :, :]
                for t in range(n_times):
                    corr[n_c, t] = fcn(x_c[:, :, t], y_c[:, :, t])

        return corr
    return correlate


def get_distance_correlation(implementation='auto'):
    """Get the function to compute the distance correlation.

    Parameters
    ----------
    implementation : {'auto', 'frites', 'dcor'}
        description
    """
    if implementation == 'dcor':
        logger.debug('Using dcor implementation of dcorr')
        from dcor import distance_correlation as dcorr
        return dcorr, 'dcor'
    elif implementation == 'frites':
        logger.debug('Using home-made implementation of dcorr')
        return distance_correlation, 'frites'
    elif implementation == 'auto':
        try:
            logger.debug('Using dcor implementation of dcorr')
            from dcor import distance_correlation as dcorr
            return dcorr, 'dcor'
        except ModuleNotFoundError:
            logger.debug('Using home-made implementation of dcorr')
            return distance_correlation, 'frites'

###############################################################################
###############################################################################
#                            DISTANCE CORRELATION
###############################################################################
###############################################################################


def dist_eucl(x):
    """Double centered euclidian distance."""
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n = x.shape[0]

    # compute the euclidian distance
    dist = - 2 * x.dot(x.T)
    x_square = (x * x).sum(axis=1)
    np.add(dist, x_square.reshape(n, 1), out=dist)
    np.add(dist, x_square.reshape(1, n), out=dist)
    np.fill_diagonal(dist, 0.)
    np.sqrt(dist, out=dist)

    # double centering
    np.subtract(dist, dist.mean(axis=0, keepdims=True), out=dist)
    np.subtract(dist, dist.mean(axis=1, keepdims=True), out=dist)
    np.add(dist, dist.mean(), out=dist)

    return dist


def distance_correlation(x, y):
    """Compute the distance correlation.

    This function computes the distance correlation between two, possibly
    multivariate, variables.

    Parameter
    ---------
    x, y : array_like
        Arrays of shape (n_samples, n_var)

    Returns
    -------
    dcorr : float
        The distance correlation between x and y
    """
    # inputs checking
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    if x.dtype not in [np.float32, np.float64]:
        x = x.astype(np.float32, copy=False)
    if y.dtype not in [np.float32, np.float64]:
        y = y.astype(np.float32, copy=False)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    assert (x.ndim == 2) and (y.ndim == 2)
    assert (x.shape[0] == y.shape[0])

    # compute distance across multivariate axis
    n = x.shape[0]
    a = dist_eucl(x)
    b = dist_eucl(y)

    # compute covariances
    denom = float(n * n)
    dcov2_xy = (a * b).sum() / denom
    dcov2_xx = (a * a).sum() / denom
    dcov2_yy = (b * b).sum() / denom
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor
