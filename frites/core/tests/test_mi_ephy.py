"""Test functions for computing mi on electrophysiological data."""
import numpy as np

from frites.core.mi_bin_ephy import (entropy, histogram, histogram2d, mi_bin,
                                     mi_bin_ccd, mi_bin_time, mi_bin_ccd_time)

class TestMiBin(object):   # noqa

    def test_entropy(self):
        """Test function entropy."""
        x = np.random.rand(1000).astype(np.float32)
        f = entropy(x)
        assert isinstance(f, float)

    def test_histogram(self):
        """Test function histogram."""
        x = np.random.rand(1000).astype(np.float32)
        h = histogram(x, 8)
        assert isinstance(h, np.ndarray) and (len(h) == 8)

    def test_histogram2d(self):
        """Test function histogram2d."""
        x = np.random.rand(1000).astype(np.float32)
        y = np.random.rand(1000).astype(np.float32)
        h = histogram2d(x, y, 8, 16)
        assert isinstance(h, np.ndarray) and (h.shape == (8, 16))

    def test_mi_bin(self):
        """Test function mi_bin."""
        x = np.random.rand(1000).astype(np.float32)
        y = np.random.rand(1000).astype(np.float32)
        mi = mi_bin(x, y, 8, 8)
        assert isinstance(mi, float)

    def test_mi_bin_ccd(self):
        """Test function mi_bin_ccd."""
        x = np.random.rand(1000).astype(np.float32)
        y = np.random.rand(1000).astype(np.float32)
        z = np.array([0] * 500 + [1] * 500).astype(np.float32)
        mi = mi_bin_ccd(x, y, z, 8)
        assert isinstance(mi, float)

    def test_mi_bin_time(self):
        """Test function mi_bin_time."""
        x = np.random.rand(10, 1000).astype(np.float32)
        y = np.random.rand(1000).astype(np.float32)
        mi = mi_bin_time(x, y, 8, 8)
        assert isinstance(mi, np.ndarray) and (len(mi) == 10)

    def test_mi_bin_ccd_time(self):
        """Test function mi_bin_ccd_time."""
        x = np.random.rand(10, 1000).astype(np.float32)
        y = np.random.rand(1000).astype(np.float32)
        z = np.array([0] * 500 + [1] * 500).astype(np.float32)
        mi = mi_bin_ccd_time(x, y, z, 8)
        assert isinstance(mi, np.ndarray) and (len(mi) == 10)

if __name__ == '__main__':
    TestMiBin().test_mi_bin_ccd_time()
