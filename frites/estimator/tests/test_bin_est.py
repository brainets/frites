"""Test the Binning-based estimator."""
import numpy as np

from frites.estimator import BinMIEstimator

np.random.seed(0)

n_times = 40
n_mv = 2
n_samples = 100
x = np.random.rand(n_times, n_mv, n_samples)
y_f = np.random.normal(size=(n_samples))
y_i = np.random.randint(0, 2, size=(n_samples))
z = np.random.randint(0, 2, size=(n_samples))
categories = np.array([0] * 40 + [1] * 30 + [0] * 30)
# effect definition
sl_effect = slice(15, 25)
effect = np.zeros((n_times,))
effect[sl_effect] = 1.


class TestBinMIEstimator(object):

    @staticmethod
    def _compare_effects(effect, mi):
        """Compare ground-truth effect with obtained MI."""
        mi = mi.squeeze()
        mi = (mi > (mi.min() + mi.max()) / 2).astype(float)
        perc = 100 * (effect == mi).sum() / n_times
        assert perc > 95

    def test_smoke_cc(self):
        """Smoke test for cc MI."""
        c = BinMIEstimator(mi_type='cc')
        c.estimate(x, y_f, categories=None)
        c.estimate(x, y_f, categories=categories)

    def test_smoke_cd(self):
        """Smoke test for cd MI."""
        c = BinMIEstimator(mi_type='cd')
        c.estimate(x, y_i, categories=None)
        c.estimate(x, y_i, categories=categories)

    def test_smoke_ccd(self):
        """Smoke test for ccd MI."""
        c = BinMIEstimator(mi_type='ccd')
        c.estimate(x, y_f, z=z, categories=None)
        c.estimate(x, y_f, z=z, categories=categories)

    def test_support_dim(self):
        """Test the support for different dimensions."""
        c = BinMIEstimator(mi_type='cc')
        # 1d
        x = np.random.rand(100)
        c = BinMIEstimator(mi_type='cc')
        mi = c.estimate(x, y_i)
        assert mi.shape == (1, 1)
        # 2d
        x = np.random.rand(1, 100)
        mi = c.estimate(x, y_i)
        assert mi.shape == (1, 1)
        # Nd
        x = np.random.rand(4, 5, 6, 1, 100)
        mi = c.estimate(x, y_i)
        assert mi.shape == (1, 4, 5, 6)

    def test_functional_cc(self):
        """Functional test for cc MI."""
        # build the testing data
        x = np.random.rand(n_times, n_mv, n_samples)
        y = np.random.normal(size=(n_samples))
        x[sl_effect, :, :] += y.reshape(1, 1, -1)
        # functional test
        c = BinMIEstimator(mi_type='cc')
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
        c = BinMIEstimator(mi_type='cd')
        mi = c.estimate(x, y)
        self._compare_effects(effect, mi)

    def test_functional_ccd(self):
        """Functional test for ccd MI."""
        # build the testing data
        x = np.random.rand(n_times, n_mv, n_samples)
        y_pos = np.random.normal(size=(50,))
        y_neg = np.random.normal(size=(50,))
        x[sl_effect, :, 0:50] += y_pos.reshape(1, 1, -1)
        x[sl_effect, :, 50::] -= y_neg.reshape(1, 1, -1)
        y = np.r_[y_pos, y_neg]
        z = np.array([0] * 50 + [1] * 50)
        # functional test
        c = BinMIEstimator(mi_type='ccd')
        mi = c.estimate(x, y, z=z)
        self._compare_effects(effect, mi)


if __name__ == '__main__':
    TestBinMIEstimator().test_functional_ccd()
