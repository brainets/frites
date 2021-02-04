"""Test high-level mutual information functions."""
import numpy as np

from frites.core import permute_mi_vector, permute_mi_trials

rnd = np.random.RandomState(0)

mi_types = ['cc', 'cd', 'ccd']
inferences = ['ffx', 'rfx']
n_perm = 15

n_times, n_epochs, n_suj, n_conds = 100, 100, 2, 3
y = rnd.rand(n_epochs)
suj = np.round(np.linspace(0, n_suj, n_epochs)).astype(int)


class TestMiStats(object):  # noqa

    def test_permute_mi_vector(self):
        """Test function permute_mi_vector."""
        for mi in mi_types:
            for inf in inferences:
                y_p = permute_mi_vector(y, suj, mi_type=mi, inference=inf,
                                        n_perm=n_perm)
                assert len(y_p) == n_perm

    def test_permute_mi_trials(self):
        """Test function permute_mi_trials."""
        suj = np.array([0, 0, 0, 1, 1, 1])
        for inf in inferences:
            y_p = permute_mi_trials(suj, inference=inf, n_perm=n_perm)
            assert len(y_p) == n_perm

            # rfx testing
            if inf == 'ffx':
                assert any([k[0:3].max() > 2 for k in y_p])
            elif inf == 'rfx':
                for k in y_p:
                    assert (k[0:3].min() == 0) and (k[0:3].max() == 2)
                    assert (k[3::].min() == 3) and (k[3::].max() == 5)

if __name__ == '__main__':
    TestMiFun().test_mi_fun()
