"""Test the different supported I/O types for DatasetEphy."""
import numpy as np

from mne import EpochsArray, create_info
from xarray import DataArray
import pandas as pd

from frites.dataset.ds_ephy_io import (ds_ephy_io, mne_to_arr, xr_to_arr)


n_epochs = 5
n_roi = 3
n_times = 10
n_suj = 2

x = [np.random.rand(n_epochs, n_roi, n_times) for k in range(n_suj)]
sf = 128
times = np.arange(n_times) / sf - 1
roi = [np.array([f"roi_{i}" for i in range(n_roi)]) for _ in range(n_suj)]
y = [np.random.rand(n_epochs) for k in range(n_suj)]
z = [np.random.randint(0, 2, (n_epochs,)) for k in range(n_suj)]


class TestDsEphyIO(object):

    @staticmethod
    def _to_mne():
        x_mne = []
        for k in range(n_suj):
            info = create_info(roi[k].tolist(), sf)
            x_mne += [EpochsArray(x[k], info, tmin=times[0], verbose=False)]
        return x_mne

    @staticmethod
    def _to_xr():
        x_xr = []
        for k in range(n_suj):
            ind = pd.MultiIndex.from_arrays([y[k], z[k]], names=('y', 'z'))
            x_xr += [DataArray(x[k], dims=('epochs', 'roi', 'times'),
                               coords=(ind, roi[k], times))]
        return x_xr

    def test_mne_to_arr(self):
        """Test function mne_to_arr."""
        # extract array either with / without roi
        x_mne = self._to_mne()
        x_m_1, times_m_1, roi_m_1 = mne_to_arr(x_mne.copy(), roi=None)
        x_m_2, times_m_2, roi_m_2 = mne_to_arr(x_mne.copy(), roi=roi)
        # testing outputs
        np.testing.assert_array_equal(times_m_1, times_m_2)
        for k in range(len(x_mne)):
            np.testing.assert_array_equal(x_m_1[k], x_m_2[k])
            np.testing.assert_array_equal(roi_m_1[k], roi_m_2[k])

    def test_xr_to_arr(self):
        """Test function xr_to_arr."""
        # testing elements independantly
        x_xr = self._to_xr()
        x_x, roi_xn, y_xn, z_xn, times_xn, sub_roi = xr_to_arr(x_xr.copy())
        assert roi_xn == y_xn == z_xn == times_xn == None
        roi_x = xr_to_arr(x_xr.copy(), roi='roi')[1]
        y_x = xr_to_arr(x_xr.copy(), y='y')[2]
        z_x = xr_to_arr(x_xr.copy(), z='z')[3]
        times_x = xr_to_arr(x_xr.copy(), times='times')[4]
        # testing results
        np.testing.assert_array_equal(times, times_x)
        for k in range(n_suj):
            np.testing.assert_array_equal(x[k], x_x[k])
            np.testing.assert_array_equal(roi[k], roi_x[k])
            np.testing.assert_array_equal(y[k], y_x[k])
            np.testing.assert_array_equal(z[k], z_x[k])

    def test_ds_ephy_io(self):
        """Test function ds_ephy_io."""
        # ---------------------------------------------------------------------
        # using numpy inputs
        x_arr, y_arr, z_arr, roi_arr, times_arr, sub_roi = ds_ephy_io(
            x, roi=roi, y=y, z=z, times=times)

        # ---------------------------------------------------------------------
        # using mne inputs
        x_mne = self._to_mne()
        x_mne, y_mne, z_mne, roi_mne, times_mne, sub_roi = ds_ephy_io(
            x_mne, roi=roi, y=y, z=z, times=times)

        # ---------------------------------------------------------------------
        x_xr = self._to_xr()
        x_xr, y_xr, z_xr, roi_xr, times_xr, sub_roi = ds_ephy_io(
            x_xr, roi='roi', y='y', z='z', times='times')

        # ---------------------------------------------------------------------
        # testing outputs
        for k in range(n_suj):
            # numpy outputs
            np.testing.assert_array_equal(x[k], x_arr[k])
            np.testing.assert_array_equal(roi[k], roi_arr[k])
            np.testing.assert_array_equal(y[k], y_arr[k])
            np.testing.assert_array_equal(z[k], z_arr[k])
            # mne outputs
            np.testing.assert_array_equal(x[k], x_mne[k])
            np.testing.assert_array_equal(roi[k], roi_mne[k])
            np.testing.assert_array_equal(y[k], y_mne[k])
            np.testing.assert_array_equal(z[k], z_mne[k])
            # xarray outputs
            np.testing.assert_array_equal(x[k], x_xr[k])
            np.testing.assert_array_equal(roi[k], roi_xr[k])
            np.testing.assert_array_equal(y[k], y_xr[k])
            np.testing.assert_array_equal(z[k], z_xr[k])
