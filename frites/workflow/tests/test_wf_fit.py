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
    rnd_x = np.random.RandomState(k)
    rnd_y = np.random.RandomState(k + n_subjects + 1)
    x += [rnd_x.rand(n_epochs, n_roi, n_times)]
    y += [rnd_y.rand(n_epochs)]
    z += [np.array([0] * half_e + [1] * half_e)]
roi = [np.array([f"roi_{k}" for k in range(n_roi)])] * n_subjects


class TestWfFit(object):

    def test_definition(self):
        WfFit()

    def test_fit_cc(self):
        ds = DatasetEphy(x, y, roi=roi, times=times)
        WfFit(mi_type='cc').fit(ds, **kw_fit)

    def test_fit_cd(self):
        ds = DatasetEphy(x, y, roi=roi, times=times)
        WfFit(mi_type='cd').fit(ds, **kw_fit)

    def test_fit_ccd(self):
        ds = DatasetEphy(x, y, z=z, roi=roi, times=times)
        WfFit(mi_type='ccd').fit(ds, **kw_fit)

    def test_stats(self):
        # FFX
        ds = DatasetEphy(x, y, roi=roi, times=times)
        WfFit(mi_type='cc', inference='ffx').fit(ds, **kw_fit)
        # RFX
        ds = DatasetEphy(x, y, roi=roi, times=times)
        WfFit(mi_type='cc', inference='rfx').fit(ds, **kw_fit)

    def test_mi_methods(self):
        for meth in ['gc', 'bin']:
            ds = DatasetEphy(x, y, roi=roi, times=times)
            WfFit(mi_type='cc', inference='ffx', mi_method=meth).fit(
                ds, **kw_fit)

    def test_biunidirected(self):
        # bidirected
        ds = DatasetEphy(x, y, roi=roi, times=times)
        fd = WfFit(mi_type='cd').fit(ds, net=False, **kw_fit)[0]
        # unidirected
        ds = DatasetEphy(x, y, roi=roi, times=times)
        nd = WfFit(mi_type='cd').fit(ds, net=True, **kw_fit)[0]

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
        assert fit.shape == (n_roi, n_roi, n_t)
        assert pv.shape == (n_roi, n_roi, n_t)
        assert wf.tvalues.shape == (n_roi, n_roi, n_t)
        assert len(wf.fit_roi) == len(wf.fitp_roi) == n_pairs
        assert len(wf.mi) == len(wf.mi_p) == n_roi


if __name__ == '__main__':
    TestWfFit().test_fit_cc()
