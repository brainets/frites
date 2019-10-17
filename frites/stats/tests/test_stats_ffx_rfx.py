"""Test FFX statistics."""
import numpy as np

from frites.stats import (ffx_maxstat, ffx_fdr, ffx_bonferroni,
                          ffx_cluster_maxstat, ffx_cluster_fdr,
                          ffx_cluster_bonferroni, ffx_cluster_tfce)
from frites.stats import (rfx_cluster_ttest, rfx_cluster_ttest_tfce)

rnd = np.random.RandomState(0)

# generate some random mutual information
n_times, n_roi, n_suj, n_perm = 100, 5, 5, 100
sl = slice(20, 40)
# ground truth
gt = np.zeros((n_roi, n_times), dtype=bool)
gt[:, sl] = True

class TestFFXRFX(object):  # noqa

    @staticmethod
    def _is_equals(pv, tol=.95):
        is_signi = pv <= .05
        n = (is_signi == gt).sum() / is_signi.size
        assert n > tol

    def test_ffx(self):
        """Test ffx_* functions."""
        # generate the mutual information
        mi = rnd.uniform(0, 1, (n_roi, n_times))
        mi_p = rnd.uniform(0, 1, (n_perm, n_roi, n_times))
        mi[:, sl] += 1000
        # test ffx
        for meth in [ffx_maxstat, ffx_fdr, ffx_bonferroni, ffx_cluster_maxstat,
                     ffx_cluster_fdr, ffx_cluster_bonferroni,
                     ffx_cluster_tfce]:
            pv = meth(mi, mi_p)
            self._is_equals(pv)

    def test_rfx(self):
        """Test rfx_* functions."""
        mi, mi_p = [], []
        for k in range(n_roi):
            # true mi
            _mi = rnd.uniform(0, 1, (n_suj, n_times))
            _mi[:, sl] += 1000
            mi += [_mi]
            # permuted mi
            mi_p += [rnd.uniform(0, 1, (n_perm, n_suj, n_times))]
        for meth in [rfx_cluster_ttest, rfx_cluster_ttest_tfce]:
            pv = meth(mi, mi_p)
            self._is_equals(pv)
