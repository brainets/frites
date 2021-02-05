"""Test definition of electrophysiological datasets."""
import numpy as np

from frites.dataset import DatasetEphy
from frites.simulations import sim_multi_suj_ephy, sim_mi_cc
from frites.io import set_log_level


set_log_level(False)


class TestDatasetEphy(object):  # noqa

    @staticmethod
    def _get_data(n_subjects=5, n_times=50, n_roi=6, n_epochs=10,
                  n_sites_per_roi=1):
        data, roi, time = sim_multi_suj_ephy(
            modality="meeg", n_times=n_times, n_roi=n_roi, n_epochs=n_epochs,
            n_subjects=n_subjects, random_state=0,
            n_sites_per_roi=n_sites_per_roi)
        data = [k + 100 for k in data]
        y, _ = sim_mi_cc(data, snr=.8)
        z = [np.random.randint(0, 3, (10,)) for _ in range(len(y))]
        dt = DatasetEphy(data, y, roi, z=z, times=time)
        return dt

    def test_definition(self):
        """Test function definition."""
        # test array definition
        data, roi, time = sim_multi_suj_ephy(modality="intra", n_times=57,
                                             n_roi=5, n_sites_per_roi=7,
                                             n_epochs=10, n_subjects=5,
                                             random_state=0)
        y, _ = sim_mi_cc(data, snr=.8)
        z = [np.random.randint(0, 3, (10,)) for _ in range(len(y))]
        dt = DatasetEphy(data, y, roi, z=z, times=time)
        dt.groupby('roi')
        # test mne definition
        data, roi, time = sim_multi_suj_ephy(modality="meeg", n_times=57,
                                             n_roi=5, n_sites_per_roi=1,
                                             n_epochs=7, n_subjects=5,
                                             as_mne=True, random_state=0)
        y, _ = sim_mi_cc(data, snr=.8)
        ds = DatasetEphy(data, y, roi, times=time)

    def test_definition_as_neo(self):
        """Test function definition."""
        # test array definition
        data, roi, time = sim_multi_suj_ephy(modality="intra", n_times=57,
                                             n_roi=5, n_sites_per_roi=7,
                                             n_epochs=10, n_subjects=5,
                                             random_state=0)
        y, _ = sim_mi_cc(data, snr=.8)
        z = [np.random.randint(0, 3, (10,)) for _ in range(len(y))]
        dt = DatasetEphy(data, y, roi, z=z, times=time)
        dt.groupby('roi')
        # test mne definition
        data, roi, time = sim_multi_suj_ephy(modality="meeg", n_times=57,
                                             n_roi=5, n_sites_per_roi=1,
                                             n_epochs=7, n_subjects=5,
                                             as_mne=False, as_neo=True,
                                             random_state=0)
        y, _ = sim_mi_cc(data, snr=.8)
        ds = DatasetEphy(data, y, roi, times=time)

    def test_multiconditions(self):
        """Test multi-conditions remapping."""
        n_suj, n_epochs, n_roi, n_times = 3, 3, 1, 10
        x = [np.random.rand(n_epochs, n_roi, n_times) for k in range(n_suj)]
        y_1 = [[0, 10], [0, 10], [1, 20]]
        y_2 = [[1, 20], [0, 10], [1, 20]]
        y_3 = [[2, 30], [0, 10], [1, 20]]
        cats = [[0, 0, 1], [1, 0, 1], [2, 0, 1]]
        y = [np.array(y_1), np.array(y_2), np.array(y_3)]
        z = [np.array(y_1), np.array(y_2), np.array(y_3)]
        roi = [[f"roi-{k}" for k in range(n_roi)]] * n_suj
        y = [k.astype(float) for k in y]
        ds = DatasetEphy(x, y, roi, z=z)
        for _y, _z, _c in zip(ds.y, ds.z, cats):
            np.testing.assert_array_equal(_z, _c)

    def test_shapes(self):
        """Test function shapes."""
        dt = self._get_data()
        assert dt._groupedby == "subject"
        # shape checking before groupby
        assert len(dt.x) == len(dt.y) == len(dt.z) == 5
        n_suj = len(dt.x)
        assert all([dt.x[k].shape == (10, 6, 50) for k in range(n_suj)])
        assert all([dt.y[k].shape == (10,) for k in range(n_suj)])
        assert all([dt.z[k].shape == (10,) for k in range(n_suj)])
        # group by roi
        dt.groupby('roi')
        assert dt._groupedby == "roi"
        assert len(dt.x) == len(dt.y) == len(dt.z) == 6
        assert all([dt.x[k].shape == (50, 1, 50) for k in range(n_suj)])
        assert all([dt.y[k].shape == (50, 1) for k in range(n_suj)])
        assert all([dt.z[k].shape == (50, 1) for k in range(n_suj)])
        assert all([dt.suj_roi[k].shape == (50,) for k in range(n_suj)])
        assert all([dt.suj_roi_u[k].shape == (5,) for k in range(n_suj)])
        assert len(dt.roi_names) == 6

    def test_copnorm(self):
        """Test function copnorm."""
        dt = self._get_data()
        dt.groupby('roi')
        assert dt._copnormed is False
        # be sure that data are centered around 100
        assert 95 < np.ravel(dt.x).mean() < 105
        dt.copnorm()
        assert isinstance(dt._copnormed, str)
        assert -1 < np.ravel(dt.x).mean() < 1

    def test_properties(self):
        """Test function properties."""
        dt = self._get_data()
        assert dt.modality == "electrophysiological"
        x, y, z = dt.x, dt.y, dt.z  # noqa
        # test setting x
        try: dt.x = 2  # noqa
        except AttributeError: pass  # noqa
        # test setting y
        try: dt.y = 2  # noqa
        except AttributeError: pass  # noqa
        # test setting z
        try: dt.z = 2  # noqa
        except AttributeError: pass  # noqa
        # shape // nb_min_suj
        assert isinstance(dt.shape, str)
        dt.nb_min_suj

    def test_builtin(self):
        """Test function builtin."""
        dt = self._get_data()
        # __len__
        assert len(dt) == 5
        # __repr__
        repr(dt)
        str(dt)

    def test_slicing(self):
        """Test spatio-temporal slicing."""
        dt = self._get_data()
        dt[::2, :]
        dt[0.15:0.18, :]
        dt[:, dt.roi[0][0:2]]

    def test_savgol_filter(self):
        """Test function savgol_filter."""
        dt = self._get_data()
        dt.savgol_filter(31)

    def test_resample(self):
        """Test function resample."""
        dt = self._get_data()
        dt.resample(10)

    def test_get_connectivity_pairs(self):
        dt = self._get_data()
        dt.groupby('roi')
        pairs_nd = dt.get_connectivity_pairs(directed=False)
        pairs_d = dt.get_connectivity_pairs(directed=True)
        assert len(pairs_d[0]) > len(pairs_nd[0])
        assert len(pairs_d[1]) > len(pairs_nd[1])
