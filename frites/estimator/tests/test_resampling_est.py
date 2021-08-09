"""Test resampling estimator."""
import numpy as np

from frites.estimator import (ResamplingEstimator, GCMIEstimator,
                              DcorrEstimator, CorrEstimator)


# dataset variables
n_trials = 100
x = np.random.rand(n_trials)
y_d = np.array([0] * 50 + [1] * 50)
cat = np.array([0] * 25 + [1] * 75)
y_c = np.random.rand(n_trials)

# estimator creation
est_gc = GCMIEstimator(mi_type='cc', verbose=False)
est_gd = GCMIEstimator(mi_type='cd')
est_c = CorrEstimator()
est_d = DcorrEstimator()


class TestResamplingEstimator(object):

    def test_resampling_cc(self):
        """Test resampling between continuous variables."""
        for est in [est_gc, est_c, est_d]:
            # category free
            est_w = ResamplingEstimator(est)
            mi = est_w.estimate(x, y_c, z=y_d)
            assert mi.shape == (1, 1)

            # with categories
            mi = est_w.estimate(x, y_c, z=y_d, categories=cat)
            assert mi.shape == (2, 1)

    def test_resampling_cd(self):
        """Test resampling between a continuous and a discrete variables."""
        # category free
        est_w = ResamplingEstimator(est_gd)
        mi = est_w.estimate(x, y_d)
        assert mi.shape == (1, 1)

        # with categories
        mi = est_w.estimate(x, y_d, categories=cat)
        assert mi.shape == (2, 1)


if __name__ == '__main__':
    TestResamplingEstimator().test_resampling_cd()
