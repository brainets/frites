"""Test generation of random data."""
import numpy as np
from mne import EpochsArray

try:
    import neo
    HAS_NEO = True
except ModuleNotFoundError:
    HAS_NEO = False

from frites.simulations import (sim_single_suj_ephy, sim_multi_suj_ephy)


class TestGenerateData(object):
    """Test generation of random data."""

    def test_sim_single_suj_ephy(self):
        """Test function sim_single_suj_ephy."""
        # standard definition
        data, roi, time = sim_single_suj_ephy(modality="meeg", n_times=57,
                                              n_roi=5, n_sites_per_roi=7,
                                              n_epochs=100)
        assert data.shape == (100, int(7 * 5), 57)
        assert data.shape == (100, len(roi), len(time))
        # mne type
        data, _, _ = sim_single_suj_ephy(as_mne=True)
        assert isinstance(data, EpochsArray)
        # neo type
        if HAS_NEO:
            data, _, _ = sim_single_suj_ephy(as_neo=True)
            assert isinstance(data, neo.Block)

    def test_sim_multi_suj_ephy(self):
        """Test function sim_multi_suj_ephy."""
        # standard definition
        data, roi, time = sim_multi_suj_ephy(modality="intra", n_times=57,
                                             n_roi=5, n_sites_per_roi=7,
                                             n_epochs=7, n_subjects=5,
                                             random_state=0)
        assert len(data) == len(roi) == 5
        n_sites = [k.shape[1] for k in data]
        assert np.unique(n_sites).size != 1
        assert np.all([k.shape == (7, len(i), 57) for k, i in zip(data, roi)])
        # mne type
        data, _, _ = sim_multi_suj_ephy(n_subjects=5, as_mne=True)
        assert all([isinstance(k, EpochsArray) for k in data])
        # neo type
        if HAS_NEO:
            data, _, _ = sim_multi_suj_ephy(n_subjects=5, as_neo=True)
            assert all([isinstance(k, neo.Block) for k in data])

