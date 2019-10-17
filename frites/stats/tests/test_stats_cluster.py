"""Test cluster detection."""
import numpy as np

from frites.stats import temporal_clusters_permutation_test


rnd = np.random.RandomState(0)


class TestStatsClusters(object):  # noqa

    @staticmethod
    def _is_signi(pval, sl):
        """Cluster have to be significant in the desired window."""
        sl = [sl] if not isinstance(sl, list) else sl
        gt = np.zeros(pval.shape, dtype=bool)
        for k in sl:
            gt[:, k] = True
        is_signi = pval < .05
        np.testing.assert_array_equal(gt, is_signi)

    def test_temporal_clusters_permutation_test(self):
        """Test function temporal_clusters_permutation_test."""
        n_pts = 100
        sl_neg = slice(20, 40)
        sl_pos = slice(60, 80)
        sl_both = [sl_neg, sl_pos]
        # generate the mutual information
        mi = rnd.uniform(0, 1, (5, n_pts))
        mi_p = rnd.uniform(0, 1, (100, 5, n_pts))
        mi[:, sl_neg] -= 1000
        mi[:, sl_pos] += 1000
        # tail = -1
        pv_neg = temporal_clusters_permutation_test(mi, mi_p, th=-100, tail=-1)
        self._is_signi(pv_neg, sl_neg)
        # tail = 1
        pv_pos = temporal_clusters_permutation_test(mi, mi_p, th=100, tail=1)
        self._is_signi(pv_pos, sl_pos)
        # tail = 0
        pv_both = temporal_clusters_permutation_test(mi, mi_p, th=100, tail=0)
        self._is_signi(pv_both, sl_both)
