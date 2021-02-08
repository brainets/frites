"""Test preprocessing functions."""
import numpy as np
import xarray as xr

from frites.utils import savgol_filter, kernel_smoothing, nonsorted_unique

class TestPreproc(object):

    def test_savgol_filter(self):
        """Test function savgol_filter."""
        n_times, n_trials = 1000, 10
        x = np.random.rand(n_trials, n_times)
        h_freq = 20.
        sfreq = 256.
        times = np.arange(n_times) / sfreq
        trials = np.arange(n_trials)
        x_xr = xr.DataArray(x, dims=('trials', 'times'),
                            coords=(trials, times))
        # testing numpy savgol
        savgol_filter(x, h_freq, axis=1, sfreq=sfreq)
        # testing xarray savgol
        savgol_filter(x_xr, h_freq)
        savgol_filter(x_xr, h_freq, axis='times')

    def test_kernel_smoothing(self):
        """Test function kernel_smoothing."""
        x = np.random.rand(50, 100, 1000)
        da = xr.DataArray(x.copy(), dims=('trials', 'roi', 'times'))
        kern = np.hanning(10)
        kernel_smoothing(x, kern, axis=0)
        kernel_smoothing(x, kern, axis=1)
        kernel_smoothing(x, kern, axis=2)
        kernel_smoothing(x, kern, axis=-1)
        kernel_smoothing(da, kern, axis='trials')
        kernel_smoothing(da, kern, axis='roi')
        kernel_smoothing(da, kern, axis='times')

    def test_nonsorted_unique(self):
        """Test function nonsorted_unique."""
        a = ['r2', 'r0', 'r0', 'r1', 'r2']
        np.testing.assert_array_equal(nonsorted_unique(a), ['r2', 'r0', 'r1'])

if __name__ == '__main__':
    TestPreproc().test_kernel_smoothing()
