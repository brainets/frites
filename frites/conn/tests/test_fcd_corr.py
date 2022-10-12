"""Test conn_fcd_corr"""
import numpy as np
import xarray as xr

from frites.estimator import DcorrEstimator
from frites.conn import conn_dfc, conn_fcd_corr, define_windows

# sample data
x = np.random.rand(10, 3, 1000)
trials = np.arange(10)
roi = ['roi_1', 'roi_0', 'roi_0']
times = (np.arange(1000) - 10) / 64.
x = xr.DataArray(x, dims=('trials', 'roi', 'times'),
                 coords=(trials, roi, times))
win, _ = define_windows(times, slwin_len=.5, slwin_step=.1)
dfc = conn_dfc(x, times='times', roi='roi', win_sample=win, verbose=False)


class TestFCDCorr(object):

    def test_smoke(self):
        """Test that it's working (overall)."""
        # test on full network
        corr = conn_fcd_corr(dfc, roi='roi', times='times', verbose=False)
        assert corr.shape[0] == len(trials)
        assert corr.shape[1] == corr.shape[2] == len(win)

        # test on single time-point
        corr = conn_fcd_corr(dfc.isel(times=[0]), roi='roi', times='times',
                             verbose=False)
        assert corr.shape[0] == len(trials)
        assert corr.shape[1] == corr.shape[2] == 1

        # test internal reshaping
        corr = conn_fcd_corr(dfc.transpose('times', 'trials', 'roi'),
                             roi='roi', times='times', verbose=False)
        assert corr.shape[0] == len(trials)
        assert corr.shape[1] == corr.shape[2] == len(win)

    def test_kwargs(self):
        """Test with a custom estimator."""
        # testing estimator and tskip
        est = DcorrEstimator()
        corr = conn_fcd_corr(dfc, roi='roi', times='times', verbose=False,
                             estimator=est, tskip=10)
        assert np.nanmin(corr.data) > 0
        assert corr.shape[0] == len(trials)
        assert corr.shape[1] == corr.shape[2] == len(win[::10])

        # testing diagonal filling
        corr = conn_fcd_corr(dfc, roi='roi', times='times', fill_diagonal=-1,
                             verbose=False)
        cm = corr.mean('trials')
        np.testing.assert_array_equal(np.diag(cm), np.full((len(win),), -1))
