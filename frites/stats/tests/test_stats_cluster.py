"""Test cluster detection."""
import numpy as np

from frites.stats import temporal_clusters_permutation_test, cluster_threshold


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
        n_pts, n_roi = 10, 2
        sl_neg = slice(2, 4)
        sl_pos = slice(6, 8)
        sl_both = [sl_neg, sl_pos]
        # generate the mutual information
        mi = rnd.uniform(0, 1, (n_roi, n_pts))
        mi_p = rnd.uniform(0, 1, (1000, n_roi, n_pts))
        mi[:, sl_neg] -= 1000
        mi[:, sl_pos] += 1000
        for mcp in ['maxstat', 'fdr', 'bonferroni']:
            # tail = -1
            pv_neg = temporal_clusters_permutation_test(
                mi, mi_p, th=-100, tail=-1, mcp=mcp)
            self._is_signi(pv_neg, sl_neg)
            # tail = 1
            pv_pos = temporal_clusters_permutation_test(
                mi, mi_p, th=100, tail=1, mcp=mcp)
            self._is_signi(pv_pos, sl_pos)
            # tail = 0
            pv_both = temporal_clusters_permutation_test(
                mi, mi_p, th=100, tail=0, mcp=mcp)
            self._is_signi(pv_both, sl_both)

    def test_cluster_threshold(self):
        """Test function cluster_threshold."""
        x = np.random.rand(10, 20)
        x_p = np.random.rand(100, 10, 20)

        for tfce in [False, True]:
            for tail in [-1, 0, 1]:
                th = cluster_threshold(x, x_p, tail=tail, tfce=tfce)
                if tfce:
                    assert isinstance(th, dict)
                    assert ('start' in th.keys()) and ('step' in th.keys())
                else:
                    assert isinstance(th, (int, float))
