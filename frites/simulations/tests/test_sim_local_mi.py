"""Test functions for simulating local mutual information."""
import numpy as np

from frites.simulations import (sim_local_cc_ss, sim_local_cc_ms,
                                sim_local_cd_ss, sim_local_cd_ms,
                                sim_local_ccd_ms, sim_local_ccd_ss)


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
