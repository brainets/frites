"""Test workflow of mutual information."""
import numpy as np

from frites.workflow import WfMi
from frites.simulations import (sim_multi_suj_ephy, sim_mi_cc, sim_mi_cd,
                                sim_mi_ccd)
from frites.dataset import DatasetEphy

from time import time as tst


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

kw_mi = dict(n_perm=n_perm, n_jobs=1)


class TestWfMi(object):  # noqa

    def test_definition(self):
        """Test workflow definition."""
        WfMi(mi_type='cc', inference='rfx')

    def test_mi_cc(self):
        """Test method fit."""
        # built the regressor
        y, gt = sim_mi_cc(x, snr=1.)
        # run workflow
        for mi_meth in ['gc', 'bin']:
            dt = DatasetEphy(x, y, roi, times=time)
            WfMi(mi_type='cc', inference='ffx', mi_method=mi_meth,
                 verbose=False).fit(dt, **kw_mi)
            WfMi(mi_type='cc', inference='rfx', mi_method=mi_meth,
                 verbose=False).fit(dt, **kw_mi)

    def test_mi_cd(self):
        """Test method fit."""
        # built the discret variable
        x_s, y, gt = sim_mi_cd(x, snr=1.)
        # run workflow
        for mi_meth in ['gc', 'bin']:
            dt = DatasetEphy(x_s, y, roi, times=time)
            WfMi(mi_type='cd', inference='ffx', mi_method=mi_meth,
                 verbose=False).fit(dt, **kw_mi)
            WfMi(mi_type='cd', inference='rfx', mi_method=mi_meth,
                 verbose=False).fit(dt, **kw_mi)

    def test_mi_ccd(self):
        """Test method fit."""
        # built the regressor and discret variables
        y, z, gt = sim_mi_ccd(x, snr=1.)
        # run workflow
        for mi_meth in ['gc', 'bin']:
            dt = DatasetEphy(x, y, roi, z=z, times=time)
            WfMi(mi_type='ccd', inference='ffx', mi_method=mi_meth,
                 verbose=False).fit(dt, **kw_mi)
            WfMi(mi_type='ccd', inference='rfx', mi_method=mi_meth,
                 verbose=False).fit(dt, **kw_mi)

    def test_output_type(self):
        """Test function output_type."""
        import pandas as pd
        from xarray import DataArray
        y, gt = sim_mi_cc(x, snr=1.)
        dt = DatasetEphy(x, y, roi, times=time)
        wf = WfMi('cc', 'ffx', verbose=False)
        # array
        mi, pv = wf.fit(dt, output_type='array', **kw_mi)
        assert all([isinstance(k, np.ndarray) for k in [mi, pv]])
        # dataframe
        mi, pv = wf.fit(dt, output_type='dataframe', **kw_mi)
        assert all([isinstance(k, pd.DataFrame) for k in [mi, pv]])
        # dataarray
        mi, pv = wf.fit(dt, output_type='dataarray', **kw_mi)
        assert all([isinstance(k, DataArray) for k in [mi, pv]])
        # test clean
        mi, mi_p = wf.mi, wf.mi_p
        assert all([isinstance(k, list) for k in [mi, mi_p]])
        wf.clean()
        assert len(wf.mi) == len(wf.mi_p) == 0

    def test_no_stat(self):
        """Test on no stats / no permutations / don't repeat computations."""
        y, gt = sim_mi_cc(x, snr=1.)
        dt = DatasetEphy(x, y, roi, times=time)
        # compute permutations but not statistics
        kernel = np.hanning(3)
        wf = WfMi('cc', 'ffx', kernel=kernel, verbose=False)
        wf.fit(dt, level='nostat', **kw_mi)
        assert len(wf.mi) == len(wf.mi_p) == n_roi
        assert len(wf.mi_p[0].shape) != 0
        # don't compute permutations nor stats
        wf = WfMi('cc', 'ffx', verbose=False)
        mi, pv = wf.fit(dt, level=None, output_type='array', **kw_mi)
        assert wf.mi_p[0].shape == (0,)
        assert pv.min() == pv.max() == 1.
        # don't compute permutations twice
        wf = WfMi('cc', 'ffx', verbose=False)
        t_start_1 = tst()
        wf.fit(dt, mcp='fdr', **kw_mi)
        t_end_1 = tst()
        t_start_2 = tst()
        wf.fit(dt, mcp='maxstat', **kw_mi)
        t_end_2 = tst()
        assert t_end_1 - t_start_1 > t_end_2 - t_start_2
