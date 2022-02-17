import numpy as np

from frites.conn.conn_tf import _tf_decomp
from frites.conn.conn_spec import conn_spec


class TestConnSpec:

    np.random.seed(0)

    n_roi, n_times, n_epochs = 4, 1000, 20
    n_edges = int(n_roi * (n_roi - 1) / 2)
    sfreq, freqs = 200, np.arange(1, 51, 1)
    n_freqs = len(freqs)
    n_cycles = freqs / 2
    times = np.arange(0, n_times // sfreq, 1 / sfreq)
    eta = np.random.normal(0, 1, size=(n_epochs, n_roi, n_times))

    def test_tf_decomp(self, ):

        # Test output shape
        for mode in ["morlet", "multitaper"]:
            out = _tf_decomp(self.eta, self.sfreq, self.freqs, mode=mode,
                             n_cycles=self.n_cycles, n_jobs=1)
            self.__assert_shape(out.shape, conn=False)

        # For multitaper test both single and array mt_bandwidth
        out1 = _tf_decomp(self.eta, self.sfreq, self.freqs, mode="multitaper",
                          n_cycles=self.n_cycles, mt_bandwidth=4, n_jobs=1)
        out2 = _tf_decomp(self.eta, self.sfreq, self.freqs, mode="multitaper",
                          n_cycles=self.n_cycles,
                          mt_bandwidth=[4] * self.n_freqs, n_jobs=1)
        np.testing.assert_array_equal(out1, out2)
        ##################################################################
        # Compare the auto-spectra with groundtruth
        ##################################################################
        for mode in ["morlet", "multitaper"]:

            # 1. Compare for stationary sinal
            x = self.__get_signal(stationary=True)

            out = _tf_decomp(x, self.sfreq, self.freqs, mode=mode,
                             n_cycles=self.n_cycles, n_jobs=1)
            out = (out * np.conj(out)).real

            if mode == "morlet":
                val, atol = 20, 2
            else:
                val, atol = 5.8, 0.35
            idx_f = self.__get_freqs_indexes(28, 32)
            actual = out.mean(axis=(0, -1))[:, idx_f].mean(1)
            np.testing.assert_allclose(
                actual, val * np.ones_like(actual), atol=atol)

            # 2. Compare for non-stationary signal
            x = self.__get_signal(stationary=False)

            out = _tf_decomp(x, self.sfreq, self.freqs, mode=mode,
                             n_cycles=self.n_cycles, n_jobs=1)
            out = (out * np.conj(out)).real

            if mode == "morlet":
                val, atol = 11, 1
            else:
                val, atol = 3.2, 0.3
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

    ##################################################################
    # Assertion private methods
    ##################################################################

    def __assert_shape(self, shape, conn=True):
        if conn:
            assert shape == (self.n_epochs, self.n_edges,
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


if __name__ == "__main__":
    """ Run the tests """
    test = TestConnSpec()
    test.test_tf_decomp()
    test.test_conn_spec()
