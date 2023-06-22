"""Test connectivity measures."""
import numpy as np
import xarray as xr

from frites.simulations import StimSpecAR

from frites.conn import (conn_covgc, conn_te, conn_dfc, conn_ccf, conn_ii,
                         conn_pid, conn_fit)


class TestConn(object):

    def test_conn_te(self):
        """Test function conn_te."""
        n_roi, n_times, n_epochs = 4, 100, 20
        trials = np.arange(n_epochs)
        roi = [f'roi_{r}' for r in range(n_roi)]
        times = np.arange(n_times)
        max_delay = 30
        x = np.random.uniform(0, 1, (n_epochs, n_roi, n_times))
        x = xr.DataArray(x, dims=('trials', 'space', 'times'),
                         coords=(trials, roi, times))
        # test across all pairs
        te = conn_te(x, roi='space', times='times', max_delay=max_delay)
        assert te.ndim == 2
        te = conn_te(x, roi='space', times='times', max_delay=max_delay,
                     return_delays=True)
        assert te.ndim == 3

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
                        conditional=True, norm=False)

    def test_conn_ccf(self):
        """Test function conn_ccf."""
        n_trials, n_roi, n_times = 20, 3, 1000
        # create coordinates
        trials = np.arange(n_trials)
        roi = [f"roi_{k}" for k in range(n_roi)]
        times = (np.arange(n_times) - 200) / 64.
        # data creation
        rnd = np.random.RandomState(0)
        x = .1 * rnd.rand(n_trials, n_roi, n_times)
        # inject relation
        bump = np.hanning(200).reshape(1, -1)
        x[:, 0, 200:400] += bump
        x[:, 1, 220:420] += bump
        x[:, 2, 150:350] += bump
        # xarray conversion
        x = xr.DataArray(x, dims=('trials', 'roi', 'times'),
                         coords=(trials, roi, times))
        # compute delayed dfc
        kw_ccf = dict(times='times', roi='roi', n_jobs=1, verbose=False)
        conn_ccf(x, normalized=False, times_as_sample=False, **kw_ccf)
        ccf = conn_ccf(x, times_as_sample=True, **kw_ccf)
        # shape and dimension checking
        assert ccf.ndim == 3
        assert ccf.dims == ('trials', 'roi', 'times')
        assert len(ccf['trials']) == len(trials)
        np.testing.assert_array_equal(ccf['trials'].data, trials)
        assert len(ccf['roi']) == 3
        # peak detection
        ccf_m = ccf.mean('trials')
        is_peaks = np.where(ccf_m == ccf_m.max('times'))
        peaks = ccf['times'].data[is_peaks[1]]
        # peak checking
        tol = 5
        assert -20 - tol <= peaks[0] <= -20 + tol
        assert 50 - tol <= peaks[1] <= 50 + tol
        assert 70 - tol <= peaks[2] <= 70 + tol

    def test_conn_ii(self):
        """Test function conn_ii."""
        # define the simulated data
        n_trials = 100
        n_roi = 4
        n_times = 1000
        y = np.random.rand(n_trials)
        roi = ['roi_1', 'roi_2', 'roi_3', 'roi_4']
        times = np.arange(n_times) / 512.
        x = np.random.rand(n_trials, n_roi, n_times)
        # introduce redundancy and synergy; Red=(0, 1), Syn=(2, 3)
        y_repeated = np.tile(y.reshape(-1, 1), (1, 100))
        y_repeated *= np.hanning(y_repeated.shape[-1]).reshape(1, -1)
        x[:, 0, 500 - 50:500 + 50] += y_repeated
        x[:, 1, 500 - 50:500 + 50] += y_repeated
        x[0:50:, 2, 500 - 50:500 + 50] += y_repeated[0:50, ...]
        x[50::, 3, 500 - 50:500 + 50] += y_repeated[50::, ...]
        x = xr.DataArray(x, dims=('trials', 'roi', 'times'),
                         coords=(y, roi, times))
        # compute ii
        ii = conn_ii(
            x, y, roi='roi', times='times', mi_type='cc', dt=10
        )
        # test that min is (0, 1) and max is (2, 3)
        ii = ii.min('times')
        assert (ii['roi'].data[
            np.where(ii.data == ii.data.min())[0]] == 'roi_1-roi_2')
        assert (ii['roi'].data[
            np.where(ii.data == ii.data.max())[0]] == 'roi_3-roi_4')

    def test_conn_pid(self):
        """Test function conn_pid."""
        ar_type = 'hga'
        n_stim = 2
        n_epochs = 100
        ss = StimSpecAR()
        ar = ss.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim)
        infotot, unique, redundancy, synergy = conn_pid(
            ar, 'trials', roi='roi', times='times', mi_type='cd', dt=10,
            verbose=False, gcrn=True
        )

    def test_conn_fit(self):
        """Test function conn_fit."""
        ar_type = 'hga'
        n_stim = 2
        n_epochs = 100
        ss = StimSpecAR()
        x = ss.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim)

        for m in ['cc', 'cd']:  # mi_type
            for n in [True, False]:  # net
                for ad in [True, False]:  # avg_delay
                    fit = conn_fit(x, y='trials', roi='roi', times='times',
                                   mi_type=m, max_delay=.3, net=n,
                                   verbose=False, avg_delay=ad)
                    print(fit.shape)
                    if n:
                        assert len(fit['roi']) == 1
                    else:
                        assert len(fit['roi']) == 2

                    if ad:
                        assert fit.ndim == 3
                    else:
                        assert fit.ndim == 2


if __name__ == '__main__':
    TestConn().test_conn_fit()
