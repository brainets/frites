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
        self._n_resampling = n_resampling

    def estimate(self, x, y, z=None, categories=None):
        fcn = self.get_function()
        return fcn(x, y, z=z, categories=categories)

    def get_function(self):

        _fcn = self._estimator.get_function()
        n_resampling = self._n_resampling

        def estimator(x, y, z=None, categories=None):
            # define how to balance the classes
            classes = y if z is None else z
            u_classes = nonsorted_unique(classes)
            assert len(u_classes) == 2, "Currently only works for 2 classes"
            n_per_classes = {u: (classes == u).sum() for u in u_classes}
            min_nb = min(n_per_classes.values())
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
