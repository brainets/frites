"""Resampling estimator."""
import numpy as np

from frites.estimator.est_mi_base import BaseMIEstimator
from frites.utils import nonsorted_unique


class ResamplingEstimator(BaseMIEstimator):

    """Trial-resampling estimator.

    In case of unbalanced contrast (i.e. when the number of trials per
    condition is very different) it can be interesting to use a
    trial-resampling technique to minimize the possibility that the effect
    size is driven by the number of trials. To this end, the same number of
    trials is used to estimate the effect size and the final

    Parameters
    ----------
    estimator : frites.estimator
        An estimator object (e.g. GCMIEstimator, CorrEstimator etc.)
    n_resampling : int | 100
        Number of resampling to perform
    """

    def __init__(self, estimator, n_resampling=100, verbose=None):
        """Init."""
        self.name = f'{estimator.name} (n_resampling={n_resampling})'
        mi_type = estimator.settings['mi_type']
        assert mi_type in ['cc', 'cd']
        super(ResamplingEstimator, self).__init__(
            mi_type=mi_type, verbose=verbose)
        # update internal settings
        settings = dict(n_resampling=n_resampling)
        self.settings.merge([settings, estimator.settings.data])
        # track internals
        self._estimator = estimator

    def estimate(self, x, y, z=None, categories=None):
        """Estimate the amount of information shared with resampling.

        This method is made for estimating information on 3D variables
        (i.e (n_var, n_mv, n_samples)) where n_var is an additional
        dimension (e.g times, times x freqs etc.), n_mv is a multivariate
        axis and n_samples the number of samples. When computing MI, both the
        multivariate and samples axes are reduced.

        In contrast to other estimators, the resampling estimator need to know
        whether the trials belong to the first or second condition. Therefore,
        a vector of discrete conditions need to be provided, depending of the
        type of information :

            * If `mi_type` is 'cc' : x and y should be continuous variables and
              z should contains the discrete classes
            * If `mi_type` is 'cd' : x should be continuous and the y variables
              is used to find the discrete classes

        Parameters
        ----------
        x : array_like
            Array of shape (n_var, n_mv, n_samples).
        y : array_like
            Array with a shape that depends on the type of MI (mi_type) :

                * If mi_type is 'cc', y should be an array with the
                  same shape as x
                * If mi_type is 'cd', y should be a row vector of shape
                  (n_samples,). In that case, the discrete values of y are used
                  to identify the discrete classes

        z : array_like | None
            Array containing discrete classes in case mi_type is 'cc'. Should
            be a vector of shape (n_samples,)

        categories : array_like | None
            Row vector of categories. This vector should have a shape of
            (n_samples,) and should contains integers describing the category
            of each sample.

        Returns
        -------
        info : array_like
            Array of information of shape (n_categories, n_var). If categories
            is None when computing information, n_categories is going to be
            one.
        """
        fcn = self.get_function()
        return fcn(x, y, z=z, categories=categories)

    def get_function(self):
        """Get the function to execute according to the input parameters.

        This can be particularly useful when computing resampling in parallel
        as it avoids to pickle the whole estimator and therefore, leading to
        faster computations.

        The returned function has the following signature :

            * fcn(x, y, z=None, categories=None)

        and return an array of shape (n_categories, n_var).
        """
        _fcn = self._estimator.get_function()
        mi_type = self.settings['mi_type']
        n_resampling = self.settings['n_resampling']

        def estimator(x, y, z=None, categories=None):
            # define which vector to use to infer discrete classes
            if mi_type == 'cd':
                classes = y.copy()
            elif mi_type == 'cc':
                classes = z.copy()

            # define how to balance the classes
            u_classes = nonsorted_unique(classes)
            assert len(u_classes) == 2, "Currently only works for 2 classes"
            n_per_classes = {u: (classes == u).sum() for u in u_classes}
            min_nb = min(n_per_classes.values())

            # generate random partitions of trials
            choices = []
            for n_k, k in enumerate(range(n_resampling)):
                _choices = []
                for c in u_classes:
                    _trials = np.where(classes == c)[0]
                    if n_per_classes[c] == min_nb:
                        _choices += [_trials]
                    else:
                        sd = np.random.RandomState(n_k)
                        __choices = sd.choice(_trials, min_nb, replace=False)
                        _choices += [__choices]
                choices += [np.concatenate(_choices)]

            # run computations
            mi = []
            for tr in choices:
                _x, _y = x[..., tr], y[..., tr]
                _cat = None if categories is None else categories[tr]
                mi.append(_fcn(_x, _y, z=tr, categories=_cat))

            # merge computations
            mi = np.stack(mi).mean(0)

            return mi

        return estimator


if __name__ == '__main__':
    from frites.estimator import GCMIEstimator

    est = GCMIEstimator(mi_type='cc')
    est_r = ResamplingEstimator(est)

    x = np.random.rand(33, 1, 400)
    y = np.random.rand(400)
    z = np.array([0] * 20 + [1] * 380)

    mi = est_r.estimate(x, y, z)

    print(mi.shape)
    print(est_r)
