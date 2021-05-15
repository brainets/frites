"""Test the GCMI estimator."""
import numpy as np

from frites.estimator import GCMIEstimator

np.random.seed(0)

n_times = 40
n_mv = 2
n_samples = 100
x = np.random.rand(n_times, n_mv, n_samples)
y_c = np.random.normal(size=(n_times, n_mv, n_samples))
y_d = np.array([0] * 50 + [1] * 50)
z_c = np.random.normal(size=(n_times, n_mv, n_samples))
z_d = np.random.randint(0, 2, (n_samples,))
categories = np.array([0] * 40 + [1] * 30 + [0] * 30)
# effect definition
sl_effect = slice(15, 25)
effect = np.zeros((n_times,))
effect[sl_effect] = 1.


class TestGCMIEstimator(object):

    @staticmethod
    def _compare_effects(effect, mi):
        """Compare ground-truth effect with obtained MI."""
        mi = mi.squeeze()
        mi = (mi > (mi.min() + mi.max()) / 2).astype(float)
        perc = 100 * (effect == mi).sum() / n_times
        assert perc > 95

    def test_smoke_cc(self):
        """Smoke test for cc MI."""
        for cop in [True, False]:
            for cat in [None, categories]:
                # tensor-based
                c = GCMIEstimator(mi_type='cc', tensor=True, copnorm=cop)
                mi_t = c.estimate(x, y_c, categories=cat)
                # vector-based
                c = GCMIEstimator(mi_type='cc', tensor=False, copnorm=cop)
                mi_v = c.estimate(x, y_c, categories=cat)
                np.testing.assert_array_almost_equal(mi_t, mi_v)

    def test_smoke_cd(self):
        """Smoke test for cd MI."""
        for cop in [True, False]:
            for cat in [None, categories]:
                # tensor-based
                c = GCMIEstimator(mi_type='cd', tensor=True, copnorm=cop)
                mi_t = c.estimate(x, y_d, categories=cat)
                # vector-based
                c = GCMIEstimator(mi_type='cd', tensor=False, copnorm=cop)
                mi_v = c.estimate(x, y_d, categories=cat)
                np.testing.assert_array_almost_equal(mi_t, mi_v)

    def test_smoke_ccd(self):
        """Smoke test for ccd MI."""
        for cop in [True, False]:
            for cat in [None, categories]:
                # tensor-based
                c = GCMIEstimator(mi_type='ccd', tensor=True, copnorm=cop)
                mi_t = c.estimate(x, y_c, z=z_d, categories=cat)
                # vector-based
                c = GCMIEstimator(mi_type='ccd', tensor=False, copnorm=cop)
                mi_v = c.estimate(x, y_c, z=z_d, categories=cat)
                np.testing.assert_array_almost_equal(mi_t, mi_v)

    def test_smoke_ccc(self):
        """Smoke test for ccc MI."""
        for cop in [True, False]:
            for cat in [None, categories]:
                # tensor-based
                c = GCMIEstimator(mi_type='ccc', tensor=True, copnorm=cop)
                mi_t = c.estimate(x, y_c, z=z_c, categories=cat)
                # vector-based
                c = GCMIEstimator(mi_type='ccc', tensor=False, copnorm=cop)
                mi_v = c.estimate(x, y_c, z=z_c, categories=cat)
                np.testing.assert_array_almost_equal(mi_t, mi_v)

    def test_support_dim(self):
        """Test the support for different dimensions."""
        y = np.random.rand(100)
        c = GCMIEstimator(mi_type='cc')
        # 1d
        x = np.random.rand(100)
        c = GCMIEstimator(mi_type='cc')
        mi = c.estimate(x, y)
        assert mi.shape == (1, 1)
        # 2d
        x = np.random.rand(1, 100)
        mi = c.estimate(x, y)
        assert mi.shape == (1, 1)
        # Nd
        x = np.random.rand(4, 5, 6, 1, 100)
        mi = c.estimate(x, y)
        assert mi.shape == (1, 4, 5, 6)

    def test_functional_cc(self):
        """Functional test for cc MI."""
        # build the testing data
        x = np.random.rand(n_times, n_mv, n_samples)
        y = np.random.normal(size=(1, 1, n_samples,))
        x[sl_effect, :, :] += y
        y = np.tile(y, (n_times, 1, 1))
        # functional test
        for cop in [True, False]:
            for imp in [True, False]:
                c = GCMIEstimator(mi_type='cc', tensor=imp, copnorm=cop)
                mi = c.estimate(x, y)
                self._compare_effects(effect, mi)

    def test_functional_cd(self):
        """Functional test for cd MI."""
        # build the testing data
        x = np.random.rand(n_times, n_mv, n_samples)
        x[sl_effect, :, 0:50] += 10.
        x[sl_effect, :, 50::] -= 10.
        y = np.array([0] * 50 + [1] * 50)
        # functional test
        for cop in [True, False]:
            for imp in [True, False]:
                c = GCMIEstimator(mi_type='cd', tensor=imp, copnorm=cop)
                mi = c.estimate(x, y)
                self._compare_effects(effect, mi)

    def test_functional_ccd(self):
        """Functional test for ccd MI."""
        # build the testing data
        x = np.random.rand(n_times, n_mv, n_samples)
        y_pos = np.random.normal(size=(1, 1, 50))
        y_neg = np.random.normal(size=(1, 1, 50))
        x[sl_effect, :, 0:50] += y_pos
        x[sl_effect, :, 50::] -= y_neg
        y = np.tile(np.concatenate((y_pos, y_neg), axis=2), (n_times, 1, 1))
        z = np.array([0] * 50 + [1] * 50)
        # functional test
        for cop in [True, False]:
            for imp in [True, False]:
                c = GCMIEstimator(mi_type='ccd', tensor=imp, copnorm=cop)
                mi = c.estimate(x, y, z=z)
                self._compare_effects(effect, mi)

    def test_functional_ccc(self):
        """Functional test for ccc MI."""
        # build the testing data
        x = np.random.rand(n_times, n_mv, n_samples)
        y = np.random.normal(size=(1, 1, n_samples,))
        x[sl_effect, :, :] += y
        y = np.tile(y, (n_times, 1, 1))
        z = np.random.rand(n_times, n_mv, n_samples)
        # functional test
        for cop in [True, False]:
            for imp in [True, False]:
                c = GCMIEstimator(mi_type='ccc', tensor=imp, copnorm=cop)
                mi = c.estimate(x, y, z=z)
                self._compare_effects(effect, mi)


if __name__ == '__main__':
    TestGCMIEstimator().test_support_dim()
