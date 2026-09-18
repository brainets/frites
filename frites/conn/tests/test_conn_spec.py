import numpy as np
import xarray as xr

from mne.time_frequency import tfr_array_morlet, tfr_array_multitaper

from frites.conn.conn_tf import _tf_decomp, _create_kernel
from frites.conn.conn_spec import conn_spec


class TestConnSpec:

    np.random.seed(0)

    n_roi, n_times, n_epochs = 4, 1000, 20
    n_edges = int(n_roi * (n_roi - 1) / 2)
    sfreq, freqs = 200, np.arange(1, 51, 1)
    n_freqs = len(freqs)
    n_cycles = freqs / 2
    # MNE uses floor(time_bandwidth - 1) tapers (default time_bandwidth=4)
    n_tapers = 3
    times = np.arange(0, n_times // sfreq, 1 / sfreq)
    eta = np.random.normal(0, 1, size=(n_epochs, n_roi, n_times))

    def test_tf_decomp(self, ):

        # Test output shape (the taper axis is kept in multitaper mode)
        for mode in ["morlet", "multitaper"]:
            out = _tf_decomp(self.eta, self.sfreq, self.freqs, mode=mode,
                             n_cycles=self.n_cycles, n_jobs=1)
            self.__assert_shape(out.shape, conn=False, mode=mode)

        # For multitaper test both single and array mt_bandwidth
        out1 = _tf_decomp(self.eta, self.sfreq, self.freqs, mode="multitaper",
                          n_cycles=self.n_cycles, mt_bandwidth=4, n_jobs=1)
        out2 = _tf_decomp(self.eta, self.sfreq, self.freqs, mode="multitaper",
                          n_cycles=self.n_cycles,
                          mt_bandwidth=[4] * self.n_freqs, n_jobs=1)
        np.testing.assert_array_equal(out1, out2)
        # scalar n_cycles with array mt_bandwidth (broadcasted)
        out3 = _tf_decomp(self.eta, self.sfreq, self.freqs, mode="multitaper",
                          n_cycles=3., mt_bandwidth=[4] * self.n_freqs,
                          n_jobs=1)
        out4 = _tf_decomp(self.eta, self.sfreq, self.freqs, mode="multitaper",
                          n_cycles=3., mt_bandwidth=4, n_jobs=1)
        np.testing.assert_allclose(out3, out4, rtol=1e-9, atol=1e-12)
        ##################################################################
        # Compare the auto-spectra with groundtruth
        ##################################################################
        for mode in ["morlet", "multitaper"]:

            # 1. Compare for stationary sinal
            x = self.__get_signal(stationary=True)

            out = _tf_decomp(x, self.sfreq, self.freqs, mode=mode,
                             n_cycles=self.n_cycles, n_jobs=1)
            out = self.__power(out, mode)

            if mode == "morlet":
                val, atol = 20, 2
            else:
                val, atol = 16.6, 1
            idx_f = self.__get_freqs_indexes(28, 32)
            actual = out.mean(axis=(0, -1))[:, idx_f].mean(1)
            np.testing.assert_allclose(
                actual, val * np.ones_like(actual), atol=atol)

            # 2. Compare for non-stationary signal
            x = self.__get_signal(stationary=False)

            out = _tf_decomp(x, self.sfreq, self.freqs, mode=mode,
                             n_cycles=self.n_cycles, n_jobs=1)
            out = self.__power(out, mode)

            if mode == "morlet":
                val, atol = 11, 1
            else:
                val, atol = 9.2, 0.6
            actual1 = out.mean(
                axis=(0, -1))[:, self.__get_freqs_indexes(8, 12)].mean(1)
            actual2 = out.mean(
                axis=(0, -1))[:, self.__get_freqs_indexes(28, 32)].mean(1)
            np.testing.assert_allclose(actual1, val * np.ones_like(actual),
                                       atol=atol)
            np.testing.assert_allclose(actual2, val * np.ones_like(actual),
                                       atol=atol)

    def test_conn_spec(self,):
        """Test function conn_spec"""
        # General parameters for the conn_spec function
        kw = dict(sfreq=self.sfreq, freqs=self.freqs, n_jobs=1, verbose=False,
                  n_cycles=self.n_cycles, times=self.times, sm_kernel='square')

        for method in ['coh', 'plv']:
            ##################################################################
            # Check general attributes of the conn_spec container
            ##################################################################
            # Compute coherence for white noise
            out = conn_spec(self.eta, sm_times=2., metric=method, **kw)
            # Test container attributes, dims and coords
            assert out.name == method
            self.__assert_shape(out.shape)
            self.__assert_default_rois(out.roi.data)
            self.__assert_dims(out.dims)
            self.__assert_attrs(out.attrs)
            ##################################################################
            # Compare output with groundtruth
            ##################################################################
            # 1. Compare with spectral conn for stationary sinal
            x = self.__get_signal(stationary=True)

            out = conn_spec(x, sm_times=2., metric=method, **kw)

            actual = out.mean(dim=("trials", "times")).sel(
                freqs=slice(28, 32)).mean("freqs")
            np.testing.assert_allclose(
                actual, 0.80 * np.ones_like(actual), atol=0.1)

            # 2. Compare with no stationary signal
            x = self.__get_signal(stationary=False)

            out = conn_spec(x, sm_times=0.6, metric=method, **kw)

            actual_1 = out.mean("trials").sel(freqs=slice(8, 12),
                                              times=slice(0.5, 2.2))
            actual_2 = out.mean("trials").sel(freqs=slice(28, 33),
                                              times=slice(2.8, 4.7))
            actual_1 = actual_1.mean(dim="freqs")
            actual_2 = actual_2.mean(dim="freqs")
            if method == "coh":
                val = 0.8
            else:
                val = 0.9
            np.testing.assert_allclose(actual_1, val * np.ones_like(actual_1),
                                       atol=0.1)
            np.testing.assert_allclose(actual_2, val * np.ones_like(actual_2),
                                       atol=0.1)

    def test_multitaper_reference(self):
        """Multitaper sxy / coh / plv against a hand-written estimator.

        The reference averages the per-taper cross- and auto-spectra (not the
        complex coefficients, and not the per-taper coherence).
        """
        x, freqs, sfreq = self.__get_coupled_pair()
        n_cycles, tw = 5, 4
        kw = dict(sfreq=sfreq, freqs=freqs, sm_times=0, n_cycles=n_cycles,
                  roi='roi', times='times', verbose=False, n_jobs=1,
                  mode='multitaper', mt_bandwidth=tw)

        # reference : (trials, chans, tapers, freqs, times)
        w = tfr_array_multitaper(x.data, sfreq, freqs, n_cycles=n_cycles,
                                 time_bandwidth=tw, output='complex',
                                 verbose=False)
        s_xy = (w[:, 0] * np.conj(w[:, 1])).mean(1)
        s_xx = (np.abs(w[:, 0]) ** 2).mean(1)
        s_yy = (np.abs(w[:, 1]) ** 2).mean(1)
        coh_ref = np.abs(s_xy) ** 2 / (s_xx * s_yy)
        plv_ref = np.abs((w[:, 1] * np.conj(w[:, 0]) / np.abs(
            w[:, 0] * w[:, 1])).mean(1))

        sxy = conn_spec(x, metric='sxy', dtype=np.complex128, **kw)
        coh = conn_spec(x, metric='coh', dtype=np.float64, **kw)
        plv = conn_spec(x, metric='plv', dtype=np.float64, **kw)
        np.testing.assert_allclose(sxy.sel(roi='x-y').data, s_xy, rtol=1e-6)
        np.testing.assert_allclose(coh.sel(roi='x-y').data, coh_ref,
                                   rtol=1e-6)
        np.testing.assert_allclose(plv.sel(roi='x-y').data, plv_ref,
                                   rtol=1e-6)
        # the estimator is not degenerate : coupled pair > independent pair
        assert 0.1 < float(coh.sel(roi='x-z').mean()) < float(
            coh.sel(roi='x-y').mean()) < 0.99
        assert float(plv.max()) <= 1. + 1e-6

    def test_sxy_complex(self):
        """The cross-spectrum is complex and follows x * conj(y)."""
        x, freqs, sfreq = self.__get_coupled_pair()
        n_cycles = 5
        kw = dict(sfreq=sfreq, freqs=freqs, sm_times=0, n_cycles=n_cycles,
                  roi='roi', times='times', verbose=False, n_jobs=1)
        # default (real) dtype is promoted, imaginary part is not lost
        sxy = conn_spec(x, metric='sxy', **kw)
        assert np.issubdtype(sxy.dtype, np.complexfloating)
        assert np.abs(sxy.data.imag).max() > 0
        # morlet cross-spectrum is exactly w_x * conj(w_y)
        w = tfr_array_morlet(x.data, sfreq, freqs, n_cycles=n_cycles,
                             output='complex', verbose=False)
        sxy = conn_spec(x, metric='sxy', dtype=np.complex128, **kw)
        np.testing.assert_allclose(sxy.sel(roi='x-y').data,
                                   w[:, 0] * np.conj(w[:, 1]), rtol=1e-6)

    def test_hanning_kernel(self):
        """Small hanning kernels are neither NaN nor a delta."""
        for n in [1, 2, 3, 5]:
            k = _create_kernel(n, 1, kernel='hanning')
            assert k.shape == (1, n)
            assert np.isfinite(k).all()
            np.testing.assert_allclose(k.sum(), 1.)
            assert (k > 0).all()
        # sm_times=2 samples used to be NaN everywhere
        kw = dict(sfreq=self.sfreq, freqs=self.freqs, n_jobs=1, verbose=False,
                  n_cycles=self.n_cycles, times=self.times,
                  sm_kernel='hanning')
        out = conn_spec(self.eta, sm_times=2. / self.sfreq, metric='coh',
                        **kw)
        assert np.isfinite(out.data).all()
        # smoothed single-trial coherence of independent noise is well below 1
        out = conn_spec(self.eta, sm_times=2., metric='coh', **kw)
        assert float(out.mean()) < 0.6

    ##################################################################
    # Assertion private methods
    ##################################################################

    def __assert_shape(self, shape, conn=True, mode='morlet'):
        if conn:
            assert shape == (self.n_epochs, self.n_edges,
                             self.n_freqs, self.n_times)
        elif mode == 'multitaper':
            assert shape == (self.n_epochs, self.n_roi, self.n_tapers,
                             self.n_freqs, self.n_times)
        else:
            assert shape == (self.n_epochs, self.n_roi,
                             self.n_freqs, self.n_times)

    def __assert_dims(self, dims):
        """ Assert the name of the dims """
        np.testing.assert_array_equal(
            dims, ('trials', 'roi', 'freqs', 'times'))

    def __assert_default_rois(self, rois):
        """ Assert the name of the rois generated by Frites """
        np.testing.assert_array_equal(rois,
                                      ['roi_0-roi_1', 'roi_0-roi_2',
                                       'roi_0-roi_3', 'roi_1-roi_2',
                                       'roi_1-roi_3', 'roi_2-roi_3'])

    def __assert_attrs(self, attrs):
        """ Assert the name of the atributes of the connectivity container """
        att = ['sfreq', 'sources', 'targets', 'sm_times', 'sm_freqs',
               'sm_kernel', 'mode', 'n_cycles', 'mt_bandwidth', 'decim',
               'type']
        np.testing.assert_array_equal(list(attrs.keys()), att)

    ##################################################################
    # Utilities private methods
    ##################################################################
    def __get_signal(self, stationary=False):
        """ Return signal used in the test """
        if stationary:
            return np.sin(2 * np.pi * self.times * 30) + self.eta
        else:
            half = self.n_times / (2 * self.sfreq)
            return np.sin(2 * np.pi * self.times * 10) * (self.times < half)\
                + np.sin(2 * np.pi * self.times * 30) * (self.times >= half)\
                + self.eta

    def __get_freqs_indexes(self, f_low, f_high):
        """ Get the indexes of a range of frequencies in the freqs array """
        return np.logical_and(self.freqs >= f_low, self.freqs <= f_high)

    @staticmethod
    def __power(w, mode):
        """Power from the complex coefficients (averaged over tapers)."""
        pw = (w * np.conj(w)).real
        return pw.mean(2) if mode == 'multitaper' else pw

    @staticmethod
    def __get_coupled_pair():
        """Three channels : x and y share a lagged 20Hz component, z is
        independent noise."""
        rng = np.random.RandomState(1)
        sfreq, n_trials, n_times = 200., 30, 600
        times = np.arange(n_times) / sfreq
        s = np.sin(2 * np.pi * 20 * times)
        x = rng.randn(n_trials, 3, n_times)
        x[:, 0] += 3 * s
        x[:, 1] += 3 * np.roll(s, 3)
        x = xr.DataArray(x, dims=('trials', 'roi', 'times'),
                         coords=(np.arange(n_trials), ['x', 'y', 'z'], times))
        return x, np.array([10., 20., 30.]), sfreq


if __name__ == "__main__":
    """ Run the tests """
    test = TestConnSpec()
    test.test_tf_decomp()
    test.test_conn_spec()
    test.test_multitaper_reference()
    test.test_sxy_complex()
    test.test_hanning_kernel()
