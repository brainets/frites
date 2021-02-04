"""Test the GCMI estimator."""
import numpy as np

from frites.estimator import GCMIEstimator

np.random.seed(0)

n_times = 20
n_mv = 2
n_samples = 100
x = np.random.rand(n_times, n_mv, n_samples)
y_c = np.random.normal(size=(n_times, n_mv, n_samples))
y_d = np.array([0] * 50 + [1] * 50)
z_c = np.random.normal(size=(n_times, n_mv, n_samples))
z_d = np.random.randint(0, 2, (n_samples,))
categories = np.array([0] * 40 + [1] * 30 + [0] * 30)


class TestGCMIEstimator(object):

    def test_mi_cc(self):
        for cop in [True, False]:
            for cat in [None, categories]:
                # tensor-based
                c = GCMIEstimator(mi_type='cc', tensor=True, copnorm=cop)
                mi_t = c.estimate(x, y_c, categories=cat)
                # vector-based
                c = GCMIEstimator(mi_type='cc', tensor=False, copnorm=cop)
                mi_v = c.estimate(x, y_c, categories=cat)
                np.testing.assert_array_almost_equal(mi_t, mi_v)

    def test_mi_cd(self):
        for cop in [True, False]:
            for cat in [None, categories]:
                # tensor-based
                c = GCMIEstimator(mi_type='cd', tensor=True, copnorm=cop)
                mi_t = c.estimate(x, y_d, categories=cat)
                # vector-based
                c = GCMIEstimator(mi_type='cd', tensor=False, copnorm=cop)
                mi_v = c.estimate(x, y_d, categories=cat)
                np.testing.assert_array_almost_equal(mi_t, mi_v)

    def test_mi_ccd(self):
        for cop in [True, False]:
            for cat in [None, categories]:
                # tensor-based
                c = GCMIEstimator(mi_type='ccd', tensor=True, copnorm=cop)
                mi_t = c.estimate(x, y_c, z=z_d, categories=cat)
                # vector-based
                c = GCMIEstimator(mi_type='ccd', tensor=False, copnorm=cop)
                mi_v = c.estimate(x, y_c, z=z_d, categories=cat)
                np.testing.assert_array_almost_equal(mi_t, mi_v)

    def test_mi_ccc(self):
        for cop in [True, False]:
            for cat in [None, categories]:
                # tensor-based
                c = GCMIEstimator(mi_type='ccc', tensor=True, copnorm=cop)
                mi_t = c.estimate(x, y_c, z=z_c, categories=cat)
                # vector-based
                c = GCMIEstimator(mi_type='ccc', tensor=False, copnorm=cop)
                mi_v = c.estimate(x, y_c, z=z_c, categories=cat)
                np.testing.assert_array_almost_equal(mi_t, mi_v)
