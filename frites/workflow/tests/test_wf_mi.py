"""Test workflow of mutual information."""
import numpy as np

from frites.workflow import WfMi
from frites.simulations import (sim_multi_suj_ephy, sim_mi_cc, sim_mi_cd,
                                sim_mi_ccd)
from frites.dataset import DatasetEphy

modality = 'meeg'
n_subjects = 5
n_epochs = 30
n_times = 20
n_roi = 2
n_sites_per_roi = 1
as_mne = False
n_perm = 5
x, roi, time = sim_multi_suj_ephy(n_subjects=n_subjects, n_epochs=n_epochs,
                                  n_times=n_times, n_roi=n_roi,
                                  n_sites_per_roi=n_sites_per_roi,
                                  modality=modality, random_state=1)
time = np.arange(n_times) / 512


class TestWfMi(object):  # noqa

    def test_definition(self):
        """Test workflow definition."""
        WfMi('cc', 'rfx')

    def test_fit_cc(self):
        """Test method fit."""
        # built the regressor
        y, gt = sim_mi_cc(x, snr=1.)
        # run workflow
        for mi_meth in ['gc', 'bin']:
            dt = DatasetEphy(x, y, roi, times=time)
            WfMi(mi_type='cc', inference='ffx', mi_method=mi_meth,
                 verbose=False).fit(dt, n_perm=n_perm, stat_method='ffx_fdr')
            WfMi(mi_type='cc', inference='rfx', mi_method=mi_meth,
                 verbose=False).fit(dt, n_perm=n_perm,
                                    stat_method='rfx_cluster_ttest')

    def test_fit_cd(self):
        """Test method fit."""
        # built the discret variable
        x_s, y, gt = sim_mi_cd(x, snr=1.)
        # run workflow
        for mi_meth in ['gc', 'bin']:
            dt = DatasetEphy(x_s, y, roi, times=time)
            WfMi(mi_type='cd', inference='ffx', mi_method=mi_meth,
                 verbose=False).fit(dt, n_perm=n_perm, stat_method='ffx_fdr')
            WfMi(mi_type='cd', inference='rfx', mi_method=mi_meth,
                 verbose=False).fit(dt, n_perm=n_perm,
                                    stat_method='rfx_cluster_ttest')
        # key error testing
        try:
            wf = WfMi(mi_type='cd', verbose=False)
            wf.fit(dt, n_perm=n_perm, stat_method="eat_potatoes")
        except KeyError:
            pass

    def test_fit_ccd(self):
        """Test method fit."""
        # built the regressor and discret variables
        y, z, gt = sim_mi_ccd(x, snr=1.)
        # run workflow
        for mi_meth in ['gc', 'bin']:
            dt = DatasetEphy(x, y, roi, z=z, times=time)
            WfMi(mi_type='ccd', inference='ffx', mi_method=mi_meth,
                 verbose=False).fit(dt, n_perm=n_perm, stat_method='ffx_fdr')
            WfMi(mi_type='ccd', inference='rfx', mi_method=mi_meth,
                 verbose=False).fit(dt, n_perm=n_perm,
                                    stat_method='rfx_cluster_ttest')

    def test_output_type(self):
        """Test function output_type."""
        import pandas as pd
        from xarray import DataArray
        y, gt = sim_mi_cc(x, snr=1.)
        dt = DatasetEphy(x, y, roi, times=time)
        wf = WfMi('cc', 'ffx', verbose=False)
        # array
        mi, pv = wf.fit(dt, n_perm=n_perm, stat_method='ffx_maxstat',
                        output_type='array')
        assert all([isinstance(k, np.ndarray) for k in [mi, pv]])
        # dataframe
        mi, pv = wf.fit(dt, n_perm=n_perm, stat_method='ffx_maxstat',
                        output_type='dataframe')
        assert all([isinstance(k, pd.DataFrame) for k in [mi, pv]])
        # dataarray
        mi, pv = wf.fit(dt, n_perm=n_perm, stat_method='ffx_maxstat',
                        output_type='dataarray')
        assert all([isinstance(k, DataArray) for k in [mi, pv]])
        # test clean
        mi, mi_p = wf.mi, wf.mi_p
        assert all([isinstance(k, list) for k in [mi, mi_p]])
        wf.clean()
        assert len(wf.mi) == len(wf.mi_p) == 0
