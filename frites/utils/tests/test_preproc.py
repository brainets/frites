"""Test preprocessing functions."""
import numpy as np
import xarray as xr

from frites.utils import savgol_filter

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

if __name__ == '__main__':
    TestPreproc().test_savgol_filter()
