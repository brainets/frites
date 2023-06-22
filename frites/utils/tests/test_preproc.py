"""Test preprocessing functions."""
import numpy as np
import xarray as xr

from frites.utils import (
    savgol_filter, kernel_smoothing, nonsorted_unique, acf, time_to_sample,
    get_closest_sample, split_group
)


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

    def test_time_to_sample(self):
        """Test the time to sample conversion function."""
        sf = 1024.
        n_times = 1000
        times = np.arange(n_times) / sf

        # test straight conversion
        values = [1., .5, .25]
        val_i_1 = time_to_sample(values, times=times)
        val_i_2 = time_to_sample(values, sf=sf)
        np.testing.assert_equal(val_i_1, val_i_2)
        np.testing.assert_equal(val_i_1, [1024, 512, 256])

        # test rounding
        np.testing.assert_equal(time_to_sample(.7, sf=sf, round='lower'), 716)
        np.testing.assert_equal(time_to_sample(.7, sf=sf, round='upper'), 717)
        np.testing.assert_equal(time_to_sample(.7, sf=sf, round='closer'), 717)

    def test_get_closest_sample(self):
        """Test function to get the closest sample in a reference vector."""
        sf = 1024.
        n_times = 1000
        times = np.arange(n_times) / sf

        # straight testing
        closest, precisions = get_closest_sample(
            times, [times[4], times[122]], return_precision=True)
        np.testing.assert_equal(closest, [4, 122])
        np.testing.assert_equal(precisions, [0, 0])

        # test precision
        closest = get_closest_sample(times, [0.1, 0.2, .4], precision=.1)

    def test_acf(self):
        """Test function acf."""
        # testing with standard numpy inputs
        x = np.random.rand(100, 3, 1000)
        assert acf(x, axis=-1).shape == x.shape
        assert acf(x, axis=-1, demean=True).shape == x.shape

        # testing with xarray inputs (string dimension name)
        trials = [0] * 50 + [1] * 50
        roi = [f'r{k}' for k in range(x.shape[1])]
        times = (np.arange(x.shape[-1]) - 100) / 64.
        x = xr.DataArray(x, dims=('trials', 'roi', 'times'),
                         coords=(trials, roi, times))
        corr = acf(x, axis='times')
        assert corr.dims == x.dims
        assert corr['times'].data[0] == 0.
        assert corr.shape == x.shape

        # same with integer dimension
        corr = acf(x, axis=-1)
        assert corr.dims == x.dims
        assert corr['times'].data[0] == 0.
        assert corr.shape == x.shape

    def test_split_group(self):
        """Test function split_group."""
        n_suj = 4
        n_times = 100
        times = np.linspace(-.5, 1.5, n_times)
        x = []
        for i in range(n_suj):
            _x = np.random.rand(np.random.randint(1, 10, (1,))[0], n_times)
            roi = np.array([f"roi_{r}" for r in range(_x.shape[0])])
            if len(roi) > 1:
                roi[1] = roi[0]
            _x = xr.DataArray(
                _x, dims=['space', 'times'], coords=(roi, times)
            )
            x.append(_x)

        x_split, u_roi = split_group(x, axis='space')
        x_back, u_suj = split_group(x_split, names=u_roi, axis='subjects',
                                    new_axis='roi')
        for i in range(n_suj):
            np.testing.assert_array_equal(x[i], x_back[i])


if __name__ == '__main__':
    TestPreproc().test_split_group()
