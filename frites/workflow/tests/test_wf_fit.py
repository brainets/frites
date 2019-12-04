"""Test the workflow of Feature Specific Information Transfer (FIT)."""
import numpy as np

from frites.dataset import DatasetEphy
from frites.workflow import WfFit

n_subjects = 5
n_epochs = 30
n_times = 40
n_roi = 4
n_perm = 10
times = np.linspace(-1, 1, n_times)
half_e = int(n_epochs / 2.)
kw_fit = dict(n_perm=n_perm, n_jobs=1, max_delay=.1)

x, y, z = [], [], []
for k in range(n_subjects):
    x += [np.random.rand(n_epochs, n_roi, n_times)]
    y += [np.random.rand(n_epochs)]
    z += [np.array([0] * half_e + [1] * half_e)]
roi = [np.array([f"roi_{k}" for k in range(n_roi)])] * n_subjects


class TestWfFit(object):

    def test_definition(self):
        WfFit()

    def test_fit_cc(self):
        ds = DatasetEphy(x, y, roi=roi, times=times)
        wf = WfFit(mi_type='cc').fit(ds, **kw_fit)

    def test_fit_cd(self):
        ds = DatasetEphy(x, y, roi=roi, times=times)
        wf = WfFit(mi_type='cd').fit(ds, **kw_fit)

    def test_fit_ccd(self):
        ds = DatasetEphy(x, y, z=z, roi=roi, times=times)
        wf = WfFit(mi_type='ccd').fit(ds, **kw_fit)

    def test_stats(self):
        # FFX
        ds = DatasetEphy(x, y, roi=roi, times=times)
        wf = WfFit(mi_type='cc', inference='ffx').fit(
            ds, stat_method='ffx_cluster_fdr', **kw_fit)
        # RFX
        ds = DatasetEphy(x, y, roi=roi, times=times)
        wf = WfFit(mi_type='cc', inference='rfx').fit(
            ds, stat_method='rfx_cluster_ttest', **kw_fit)

    def test_mi_methods(self):
        for meth in ['gc', 'bin']:
            ds = DatasetEphy(x, y, roi=roi, times=times)
            wf = WfFit(mi_type='cc', inference='ffx', mi_method=meth).fit(
                ds, stat_method='ffx_cluster_fdr', **kw_fit)

    def test_directed(self):
        # directed
        ds = DatasetEphy(x, y, roi=roi, times=times)
        fd = WfFit(mi_type='cd').fit(ds, directed=True, output_type='2d_array',
                                     **kw_fit)[0]
        # non-directed
        ds = DatasetEphy(x, y, roi=roi, times=times)
        nd = WfFit(mi_type='cd').fit(ds, directed=False, output_type='2d_array',
                                     **kw_fit)[0]
        assert fd.shape[1] > nd.shape[1]

    def test_output_type(self):
        import pandas as pd
        from xarray import DataArray
        outs = {'2d_array': np.ndarray, '3d_array': np.ndarray,
                '2d_dataframe': pd.DataFrame, '3d_dataframe': pd.DataFrame,
                'dataarray': DataArray}

        ds = DatasetEphy(x, y, roi=roi, times=times)
        for o_type, t_type in outs.items():
            wf = WfFit(mi_type='cd')
            fit, pv = wf.fit(ds, output_type=o_type, **kw_fit)
            tv = wf.tvalues
            assert isinstance(fit, t_type)
            assert isinstance(pv, t_type)
            assert isinstance(tv, t_type)

    def test_properties(self):
        ds = DatasetEphy(x, y, roi=roi, times=times)
        # properties when worflow defined
        wf = WfFit()
        assert all([k is None for k in [wf.sources, wf.targets, wf.tvalues]])
        assert len(wf.fit_roi) == len(wf.fitp_roi) == 0
        assert len(wf.mi) == len(wf.mi_p) == 0
        # properties when worflow launched
        fit, pv = wf.fit(ds, **kw_fit)
        n_pairs = n_roi * (n_roi - 1)
        n_t = len(wf.times)
        assert len(wf.sources) == len(wf.targets) == n_pairs
        assert fit.shape == (n_t, n_pairs)
        assert pv.shape == (n_t, n_pairs)
        assert wf.tvalues.shape == (n_t, n_pairs)
        assert len(wf.fit_roi) == len(wf.fitp_roi) == n_pairs
        assert len(wf.mi) == len(wf.mi_p)  == n_roi
