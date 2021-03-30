"""Test connectivity measures."""
import numpy as np
import xarray as xr

from frites.conn import (conn_covgc, conn_transfer_entropy, conn_dfc)


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

    def test_conn_dfc(self):
        """Test function conn_dfc."""
        n_epochs = 5
        n_times = 100
        n_roi = 3
        times = np.linspace(-1, 1, n_times)
        win_sample = np.array([[10, 20], [30, 40]])
        roi = [f"roi_{k}" for k in range(n_roi)]
        x = np.random.rand(n_epochs, n_roi, n_times)

        dfc = conn_dfc(x, win_sample, times=times, roi=roi)
        assert dfc.shape == (n_epochs, 3, 2)
        dfc = conn_dfc(x, win_sample, times=times, roi=roi)
        assert isinstance(dfc, xr.DataArray)

        # test empty window definition + sorted channel aggregation
        x = np.random.rand(10, 3, 100)
        trials = np.arange(10)
        roi = ['roi_1', 'roi_0', 'roi_0']
        times = (np.arange(100) - 10) / 64.
        x = xr.DataArray(x, dims=('trials', 'roi', 'times'),
                         coords=(trials, roi, times))
        dfc = conn_dfc(x, times='times', roi='roi', agg_ch=False)
        assert dfc.shape == (10, 3, 1)
        np.testing.assert_array_equal(
            dfc['roi'].data, ['roi_0-roi_1', 'roi_0-roi_1', 'roi_0-roi_0'])

        dfc = conn_dfc(x, times='times', roi='roi', agg_ch=True)
        assert dfc.shape == (10, 1, 1)
        np.testing.assert_array_equal(dfc['roi'].data, ['roi_0-roi_1'])

    def test_conn_covgc(self):
        """Test function conn_covgc."""
        n_epochs = 5
        n_times = 100
        n_roi = 3
        x = np.random.rand(n_epochs, n_roi, n_times)
        dt = 10
        lag = 2
        t0 = [50, 80]

        _ = conn_covgc(x, dt, lag, t0, n_jobs=1, method='gc')
        gc = conn_covgc(x, dt, lag, t0, n_jobs=1, method='gauss')
        assert gc.shape == (n_epochs, 3, len(t0), 3)
        assert isinstance(gc, xr.DataArray)
        gc = conn_covgc(x, dt, lag, t0, n_jobs=1, method='gc',
                        conditional=True)
