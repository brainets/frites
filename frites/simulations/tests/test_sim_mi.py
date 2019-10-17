"""Test functions to generate random mi."""
from frites.simulations import sim_multi_suj_ephy
from frites.simulations import sim_mi_cc, sim_mi_cd, sim_mi_ccd


modality = 'intra'
n_subjects = 4
n_epochs = 10
n_times = 50
n_roi = 10
n_sites_per_roi = 1
as_mne = False
x, roi, time = sim_multi_suj_ephy(n_subjects=n_subjects, n_epochs=n_epochs,
                                  n_times=n_times, n_roi=n_roi,
                                  n_sites_per_roi=n_sites_per_roi,
                                  as_mne=as_mne, modality=modality,
                                  random_state=1)

class TestSimMi(object):  # noqa

    def test_sim_mi_cc(self):
        """Test function sim_mi_cc."""
        y, gt = sim_mi_cc(x, snr=.8)
        assert len(y) == len(x)
        assert all([k.shape == (n_epochs,) for k in y])
        assert (len(gt) == n_times) and (gt.dtype == bool)

    def test_sim_mi_cd(self):
        """Test function sim_mi_cd."""
        _, y, gt = sim_mi_cd(x, snr=.8)
        assert len(y) == len(x)
        assert all([k.shape == (n_epochs,) for k in y])
        assert (len(gt) == n_times) and (gt.dtype == bool)

    def test_sim_mi_ccd(self):
        """Test function sim_mi_ccd."""
        y, z, gt = sim_mi_ccd(x, snr=.8)
        assert len(y) == len(x) == len(z)
        assert all([k.shape == (n_epochs,) for k in y])
        assert (len(gt) == n_times) and (gt.dtype == bool)


if __name__ == '__main__':
  TestSimMi().test_sim_mi_ccd()