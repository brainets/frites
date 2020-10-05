"""Test connectivity measures."""
import numpy as np
import xarray as xr

from frites.conn import (conn_covgc, conn_transfer_entropy, conn_dfc, conn_fit,
                         conn_reshape_undirected, conn_reshape_directed)


class TestConn(object):

    def test_conn_transfer_entropy(self):
        """Test function conn_transfer_entropy."""
        n_roi, n_times, n_epochs = 4, 100, 20
        max_delay = 30
        x = np.random.uniform(0, 1, (n_roi, n_times, n_epochs))
        # test across all pairs
        te, pairs = conn_transfer_entropy(x, max_delay=max_delay)
        assert te.shape == (pairs.shape[0], n_times - max_delay)
        assert pairs.shape == (n_roi * (n_roi - 1), 2)
        # test specific pairs
        pairs = np.c_[np.array([0, 1]), np.array([2, 3])]
        n_pairs = pairs.shape[0]
        te, pairs = conn_transfer_entropy(x, max_delay=max_delay, pairs=pairs)
        assert te.shape == (n_pairs, n_times - max_delay)
        assert pairs.shape == (n_pairs, 2)

    def test_conn_fit(self):
        """Test function conn_fit."""
        n_times = 100
        max_delay = np.float32(.1)
        times = np.linspace(-1, 1, n_times).astype(np.float32)
        x_s = np.random.rand(5, 10, n_times).astype(np.float32)
        x_t = np.random.rand(5, 10, n_times).astype(np.float32)
        conn_fit(x_s, x_t, times, max_delay)

    def test_conn_dfc(self):
        """Test function conn_dfc."""
        n_epochs = 5
        n_times = 100
        n_roi = 3
        times = np.linspace(-1, 1, n_times)
        win_sample = np.array([[10, 20], [30, 40]])
        roi = [f"roi_{k}" for k in range(n_roi)]
        x = np.random.rand(n_epochs, n_roi, n_times)

        dfc = conn_dfc(x, times, roi, win_sample)[0]
        assert dfc.shape == (n_epochs, 3, 2)
        dfc = conn_dfc(x, times, roi, win_sample)[0]
        assert isinstance(dfc, xr.DataArray)

    def test_conn_covgc(self):
        """Test function conn_covgc."""
        n_epochs = 5
        n_times = 100
        n_roi = 3
        x = np.random.rand(n_epochs, n_roi, n_times)
        dt = 10
        lag = 2
        t0 = [50, 80]

        _ = conn_covgc(x, dt, lag, t0, n_jobs=1, method='gc')[0]
        gc = conn_covgc(x, dt, lag, t0, n_jobs=1, method='gauss')[0]
        assert gc.shape == (n_epochs, 3, len(t0), 3)
        assert isinstance(gc, xr.DataArray)
        gc = conn_covgc(x, dt, lag, t0, n_jobs=1, method='gc',
                        conditional=True)[0]

    def test_conn_reshape_undirected(self):
        """Test function conn_reshape_undirected."""
        # compute DFC
        n_epochs, n_times, n_roi = 5, 100, 3
        times = np.linspace(-1, 1, n_times)
        win_sample = np.array([[10, 20], [30, 40]])
        roi = [f"roi_{k}" for k in range(n_roi)]
        order = ['roi_2', 'roi_1']
        x = np.random.rand(n_epochs, n_roi, n_times)
        dfc = conn_dfc(x, times, roi, win_sample)[0].mean('trials')
        # reshape it without the time dimension
        dfc_mean = conn_reshape_undirected(dfc.mean('times'))
        assert dfc_mean.shape == (n_roi, n_roi, 1)
        # reshape it with the time dimension
        dfc_times = conn_reshape_undirected(dfc.copy())
        assert dfc_times.shape == (n_roi, n_roi, len(dfc['times']))
        # try the reorder
        dfc_order = conn_reshape_undirected(dfc, order=order)
        assert dfc_order.shape == (2, 2, len(dfc['times']))
        assert np.array_equal(dfc_order['sources'], dfc_order['targets'])
        assert np.array_equal(dfc_order['sources'], order)

    def test_conn_reshape_directed(self):
        """Test function conn_reshape_directed."""
        n_epochs, n_times, n_roi = 5, 100, 3
        x = np.random.rand(n_epochs, n_roi, n_times)
        dt, lag, t0 = 10, 2, [50, 80]
        order = ['roi_2', 'roi_1']
        # compute covgc
        gc = conn_covgc(x, dt, lag, t0, n_jobs=1, method='gauss')[0]
        gc = gc.mean('trials')
        # reshape it without the time dimension
        gc_mean = conn_reshape_directed(gc.copy().mean('times'))
        assert gc_mean.shape == (n_roi, n_roi, 1)
        # reshape it with the time dimension
        gc_times = conn_reshape_directed(gc.copy())
        assert gc_times.shape == (n_roi, n_roi, len(gc['times']))
        # try the reorder
        gc_order = conn_reshape_directed(gc.copy(), order=order)
        assert gc_order.shape == (2, 2, len(gc['times']))
        assert np.array_equal(gc_order['sources'], gc_order['targets'])
        assert np.array_equal(gc_order['sources'], order)
