"""Test high-level mutual information functions."""
import numpy as np

from frites.core import MI_FUN

rnd = np.random.RandomState(0)

mi_types = ['cc', 'cd', 'ccd']
inferences = ['ffx', 'rfx']

n_times, n_epochs, n_suj, n_conds = 100, 100, 2, 3
x = rnd.rand(n_times, 1, n_epochs)
y = rnd.rand(n_epochs)
z = np.round(np.linspace(0, n_conds, n_epochs)).astype(int)
suj = np.round(np.linspace(0, n_suj, n_epochs)).astype(int)

class TestMiFun(object):  # noqa

    def test_mi_fun(self):
        for mi in mi_types:
            for inf in inferences:
                fun = MI_FUN[mi][inf]
                y_c = z if mi is 'cd' else y
                fun(x, y_c, z, suj)        
