"""Wrapper for custom estimators."""
import numpy as np

from frites.estimator.est_mi_base import BaseMIEstimator
from frites.io import set_log_level, logger
from frites.utils import nonsorted_unique


class CustomEstimator(BaseMIEstimator):

    """Wrapper for defining custom estimator of information.

    Parameters
    ----------
    name : str
        Estimator name
    mi_type : {'cc', 'cd'}
        Specify whether the estimator is able to compute information shared
        between two continuous variables (i.e. regression like 'cc') or
        between a continuous and a discrete variables (i.e. decoding like 'cd')
    core_fun : function
        The core function to use for estimating information. The function
        should takes as input two variables (def fcn(x, y)), each of shape
        (n_var, n_mv, n_samples) where n_samples (or n_epochs, n_trials) is the
        number of repetitions along which to compute information, n_mv is a
        multivariate axis and n_var is an additional dimension. The function
        should returns the amount of information shared between x and y and the
        resulting array should have a shape of (n_var,)
    multivariate : bool | True
        Specify whether the estimator is supporting multivariate inputs (True)
        or not (False)
    test : bool | True
        Test if the provided core function is valid or not.
    """

    def __init__(self, name, mi_type, core_fun, multivariate=True, test=True,
                 verbose=None):
        """Init."""
        # initialize the estimator
        self.name = name
        assert mi_type in ['cc', 'cd']
        super(CustomEstimator, self).__init__(mi_type=mi_type, verbose=verbose)
        self._core_fun = core_fun
        self.settings.update(dict(mi_type=mi_type, multivariate=multivariate,
                                  tested=test))

        # test the final custom estimator
        if test:
            self._test_custom_estimator(
                multivariate=multivariate, verbose=verbose)

    def _test_custom_estimator(self, multivariate=True, verbose=None):
        """Test the definition of the custom estimator."""
        set_log_level(verbose)
        fcn = self._core_fun
        mi_type = self.settings['mi_type']

        # test function signature
        from inspect import signature
        sig = signature(fcn)
        if not len(sig.parameters) == 2:
            raise TypeError(f"Core function should use two inputs and not "
                            f"{sig.parameters}")
        logger.info("    Testing function's signature [PASSED]")

        # test univariate inputs
        x, y = self._test_generate_data(mi_type, univariate=True)
        mi = fcn(x, y)
        logger.info("    Testing univariate inputs [PASSED]")

        # output type
        if not isinstance(mi, np.ndarray):
            raise TypeError(f"Output type should be a NumPy array and not "
                            f"{type(mi)}")
        logger.info("    Testing univariate output type [PASSED]")

        # test univariate outputs
        if mi.shape != (10,):
            raise ValueError(
                f"When testing for inputs of shape x.shape={x.shape} and "
                f"y.shape={y.shape} outputs result in a shape of "
                f"info.shape={mi.shape} instead of info.shape=(10,)")
        logger.info("    Testing univariate outputs shape [PASSED]")

        # test multivariate
        if multivariate:
            # test multivariate inputs
            x, y = self._test_generate_data(mi_type, univariate=False)
            mi = fcn(x, y)
            logger.info("    Testing multivariate inputs [PASSED]")

            # output type
            if not isinstance(mi, np.ndarray):
                raise TypeError(f"Output type should be a NumPy array and not "
                                f"{type(mi)}")
            logger.info("    Testing multivariate output type [PASSED]")

            # test multivariate outputs
            if mi.shape != (10,):
                raise ValueError(
                    f"When testing for inputs of shape x.shape={x.shape} and "
                    f"y.shape={y.shape} outputs result in a shape of "
                    f"info.shape={mi.shape} instead of info.shape=(10,)")
            logger.info("    Testing multivariate outputs shape [PASSED]")

    @staticmethod
    def _test_generate_data(mi_type, univariate=True, n_samples=400, n_var=10):
        """Generate random data for testing estimator."""
        n_mv = 1 if univariate else 4
        shape = (n_var, n_mv, n_samples)
        x = np.random.normal(size=shape)
        if mi_type == 'cd':
            y = np.random.randint(0, high=2, size=(n_var, 1, n_samples))
        else:
            y = np.random.normal(size=shape)
        return x, y

    def estimate(self, x, y, z=None, categories=None):
        """Estimate the amount of information shared with the custom estimator.

        This method is made for estimating the amount of information shared
        between on 3D variables (i.e (n_var, n_mv, n_samples)) where n_var is
        an additional dimension (e.g times, times x freqs etc.)n_mv is a
        multivariate axis and n_samples the number of samples.

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
        info : array_like
            Array of information of shape (n_categories, n_var).
        """
        fcn = self.get_function()
        return fcn(x, y, z=z, categories=categories)

    def get_function(self):
        """Get the function to execute according to the input parameters.

        This can be particularly useful when computing information in parallel
        as it avoids to pickle the whole estimator and therefore, leading to
        faster computations.

        The returned function has the following signature :

            * fcn(x, y, *args, categories=None, **kwargs)

        and return an array of shape (n_categories, n_var).
        """
        core_fun = self._core_fun

        def estimator(x, y, z=None, categories=None):
            """Custom estimator wrapper."""
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

            # repeat y and z (if needed)
            if y.ndim == 1:
                y = y[np.newaxis, ...]
            if y.ndim == 2:
                y = y[np.newaxis, ...]
            if (y.shape[0] == 1) and (x.shape[0] != y.shape[0]):
                n_var, n_mv, _ = x.shape
                y = np.tile(y, (n_var, 1, 1))

            # compute (potentially categorical) MI
            n_var = x.shape[0]
            if isinstance(categories, np.ndarray):
                # get unique non-sorted categories
                u_cat = nonsorted_unique(categories)
                # compute per category
                mi = np.zeros((len(u_cat), n_var), dtype=x.dtype)
                for n_c, c in enumerate(u_cat):
                    is_cat = categories == c
                    mi[n_c, :] = core_fun(x[..., is_cat], y[..., is_cat])
            else:
                mi = core_fun(x, y)[np.newaxis, :]

            # retrieve original shape (if needed)
            if reshape is not None:
                mi = mi.reshape([mi.shape[0]] + reshape[0])

            return mi.astype(np.float32)

        return estimator


if __name__ == '__main__':
    x = np.random.rand(1000)
    y = np.random.rand(1000)
    cat = np.array([0] * 500 + [1] * 500)
    def fcn(x, y): return (x * y).sum((1, 2))  # noqa
    mi = CustomEstimator('test', 'cc', fcn).estimate(x, y, categories=cat)
    # print(mi.shape)
