"""Test the workflow of pairwise connectivity."""
import numpy as np

from frites.dataset import DatasetEphy
from frites.workflow import WfConn

n_subjects = 3
n_epochs = 30
n_times = 40
n_roi = 4
n_perm = 10
times = np.linspace(-1, 1, n_times)
kw_conn = dict(n_perm=n_perm, n_jobs=1)


x, y, z = [], [], []
for k in range(n_subjects):
    rnd_x = np.random.RandomState(k)
    x += [rnd_x.rand(n_epochs, n_roi, n_times)]
roi = [np.array([f"roi_{k}" for k in range(n_roi)])] * n_subjects


class TestWfConn(object):

    def test_definition(self):
        WfConn()

    def test_fit(self):
        ds = DatasetEphy(x, roi=roi, times=times)
        WfConn().fit(ds, **kw_conn)

    def test_stats(self):
        # FFX
        ds = DatasetEphy(x, roi=roi, times=times)
        WfConn(inference='ffx').fit(ds, **kw_conn)
        # RFX
        ds = DatasetEphy(x, roi=roi, times=times)
        WfConn(inference='rfx').fit(ds, **kw_conn)

    def test_mi_methods(self):
        for meth in ['gc', 'bin']:
            ds = DatasetEphy(x, roi=roi, times=times)
            WfConn(inference='ffx', mi_method=meth).fit(ds, **kw_conn)

    def test_properties(self):
        ds = DatasetEphy(x, roi=roi, times=times)
        wf = WfConn()
        wf.fit(ds, **kw_conn)
        assert isinstance(wf.mi, list)
        assert isinstance(wf.mi_p, list)
        assert all([k.shape == (n_subjects, n_times) for k in wf.mi])
        assert all([k.shape == (n_perm, n_subjects, n_times) for k in wf.mi_p])
