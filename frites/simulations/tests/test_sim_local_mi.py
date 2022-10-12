"""Test functions for simulating local mutual information."""
import numpy as np
import xarray as xr

from frites.simulations import (
    sim_local_cc_ss, sim_local_cc_ms, sim_local_cd_ss, sim_local_cd_ms,
    sim_local_ccd_ms, sim_local_ccd_ss, sim_ground_truth
)


n_subjects = 4
n_epochs = 100
n_times = 50
n_roi = 7
n_conditions = 5
cl_index = [20, 30]


class TestSimLocalMi(object):  # noqa

    def test_sim_local_cc_ss(self):
        """Test function sim_local_cc_ss."""
        x, y, roi, times = sim_local_cc_ss(
            n_epochs=n_epochs, n_roi=n_roi, n_times=n_times, cl_index=cl_index)
        assert x.shape == (n_epochs, n_roi, n_times)
        assert y.shape == (n_epochs,)
        assert times.shape == (n_times,)
        assert roi.shape == (n_roi,)

    def test_sim_local_cc_ms(self):
        """Test function sim_local_cc_ms."""
        x, y, roi, times = sim_local_cc_ms(
            n_subjects, n_epochs=n_epochs, n_roi=n_roi, n_times=n_times,
            cl_index=cl_index)
        assert len(x) == len(y) == len(roi) == n_subjects
        for k in range(n_subjects):
            assert x[k].shape == (n_epochs, n_roi, n_times)
            assert y[k].shape == (n_epochs,)
            assert roi[k].shape == (n_roi,)

    def test_sim_local_cd_ss(self):
        """Test function sim_local_cd_ss."""
        x, y, roi, times = sim_local_cd_ss(
            n_epochs=n_epochs, n_roi=n_roi, n_times=n_times, cl_index=cl_index,
            n_conditions=n_conditions)
        assert x.shape == (n_epochs, n_roi, n_times)
        assert y.shape == (n_epochs,)
        np.testing.assert_array_equal(np.unique(y), np.arange(n_conditions))
        assert times.shape == (n_times,)
        assert roi.shape == (n_roi,)

    def test_sim_local_cd_ms(self):
        """Test function sim_local_cd_ms."""
        x, y, roi, times = sim_local_cd_ms(
            n_subjects, n_epochs=n_epochs, n_roi=n_roi, n_times=n_times,
            cl_index=cl_index, n_conditions=n_conditions)
        assert len(x) == len(y) == len(roi) == n_subjects
        for k in range(n_subjects):
            assert x[k].shape == (n_epochs, n_roi, n_times)
            assert y[k].shape == (n_epochs,)
            np.testing.assert_array_equal(np.unique(y[k]),
                                          np.arange(n_conditions))
            assert roi[k].shape == (n_roi,)

    def test_sim_local_ccd_ms(self):
        """Test function sim_local_ccd_ms."""
        x, y, z, roi, times = sim_local_ccd_ss(
            n_epochs=n_epochs, n_roi=n_roi, n_times=n_times, cl_index=cl_index,
            n_conditions=n_conditions)
        assert x.shape == (n_epochs, n_roi, n_times)
        assert y.shape == z.shape == (n_epochs,)
        np.testing.assert_array_equal(np.unique(z), np.arange(n_conditions))
        assert times.shape == (n_times,)
        assert roi.shape == (n_roi,)

    def test_sim_local_ccd_ss(self):
        """Test function sim_local_ccd_ss."""
        x, y, z, roi, times = sim_local_ccd_ms(
            n_subjects, n_epochs=n_epochs, n_roi=n_roi, n_times=n_times,
            cl_index=cl_index, n_conditions=n_conditions)
        assert len(x) == len(y) == len(z) == len(roi) == n_subjects
        for k in range(n_subjects):
            assert x[k].shape == (n_epochs, n_roi, n_times)
            assert y[k].shape == z[k].shape == (n_epochs,)
            np.testing.assert_array_equal(np.unique(z[k]),
                                          np.arange(n_conditions))
            assert roi[k].shape == (n_roi,)

    def test_sim_ground_truth(self):
        """Test function sim_ground_truth."""
        n_subjects = 5
        n_epochs = 10
        kw = dict(verbose=False, random_state=0)

        for gtype in ['tri', 'tri_r', 'diffuse', 'focal']:
            # test ground truth output
            gt = sim_ground_truth(
                n_subjects, n_epochs, gtype=gtype, gt_as_cov=False,
                gt_only=True, **kw)
            assert gt.data.dtype == bool
            gt = sim_ground_truth(
                n_subjects, n_epochs, gtype=gtype, gt_as_cov=True,
                gt_only=True, **kw)
            assert gt.data.dtype == float

            # test data
            da, _ = sim_ground_truth(
                n_subjects, n_epochs, gtype=gtype, gt_as_cov=False,
                gt_only=False, **kw)
            assert len(da) == n_subjects
            assert all([isinstance(k, xr.DataArray) for k in da])
            assert all([da[0].dims == k.dims for k in da])
