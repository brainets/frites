"""Test information transfer functions."""
import numpy as np

from frites.core.it import it_transfer_entropy, it_fit


class TestIt(object):

    def test_it_transfer_entropy(self):
        """Test function it_transfer_entropy."""
        n_roi, n_times, n_epochs = 4, 100, 20
        max_delay = 30
        x = np.random.uniform(0, 1, (n_roi, n_times, n_epochs))
        # test across all pairs
        te, pairs = it_transfer_entropy(x, max_delay=max_delay)
        assert te.shape == (pairs.shape[0], n_times - max_delay)
        assert pairs.shape == (n_roi * (n_roi - 1), 2)
        # test specific pairs
        pairs = np.c_[np.array([0, 1]), np.array([2, 3])]
        n_pairs = pairs.shape[0]
        te, pairs = it_transfer_entropy(x, max_delay=max_delay, pairs=pairs)
        assert te.shape == (n_pairs, n_times - max_delay)
        assert pairs.shape == (n_pairs, 2)

    def test_it_fit(self):
        """Test function it_fit."""
        n_times = 100
        max_delay = np.float32(.1)
        times = np.linspace(-1, 1, n_times).astype(np.float32)
        x_s = np.random.rand(5, 10, n_times).astype(np.float32)
        x_t = np.random.rand(5, 10, n_times).astype(np.float32)
        it_fit(x_s, x_t, times, max_delay)
