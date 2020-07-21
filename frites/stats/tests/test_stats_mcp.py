"""Test correction of Multiple Comparison Problem (MCP)."""
import numpy as np

from frites.stats import testwise_correction_mcp as fcn_correction_mcp
from frites.stats import cluster_correction_mcp, cluster_threshold


rnd = np.random.RandomState(0)


class TestMCP(object):

    @staticmethod
    def assert_equals(tail, mcp, pv, gt, p=.05, tolerance=.05):
        is_signi = (pv < p).astype(int)
        p_correct = 1. - (is_signi == gt.astype(int)).sum() / gt.size
        assert p_correct < tolerance, (
            f"{mcp} - {tail} tail : {p_correct} >= tolerance={tolerance}")

    def test_wise_mcp_correction(self):
        """Test function testwise_correction_mcp."""
        # generate som random data
        rnd = np.random.RandomState(0)
        x = rnd.rand(8, 20)
        x_p = rnd.rand(10000, 8, 20)
        x_pos_cl, y_pos_cl = slice(1, 4), slice(6, 12)
        x_neg_cl, y_neg_cl = slice(3, 6), slice(15, 19)
        x[x_pos_cl, y_pos_cl] += 1000.
        x[x_neg_cl, y_neg_cl] -= 1000.
        p = .05
        # ground-truth
        gt_pos = np.zeros((8, 20), dtype=bool)
        gt_pos[x_pos_cl, y_pos_cl] = True
        gt_neg = np.zeros((8, 20), dtype=bool)
        gt_neg[x_neg_cl, y_neg_cl] = True
        gt_bot = np.zeros((8, 20), dtype=bool)
        gt_bot[x_pos_cl, y_pos_cl] = True
        gt_bot[x_neg_cl, y_neg_cl] = True

        # run MCP corrections
        for mcp in ['maxstat', 'fdr', 'bonferroni']:
            # positive tail
            pv_1 = fcn_correction_mcp(x, x_p, tail=1, mcp=mcp)
            self.assert_equals(1, mcp, pv_1, gt_pos, p=p)
            # negative tail
            pv_m1 = fcn_correction_mcp(x, x_p, tail=-1, mcp=mcp)
            self.assert_equals(-1, mcp, pv_m1, gt_neg, p=p)
            # both tails
            pv_2 = fcn_correction_mcp(x, x_p, tail=0, mcp=mcp)
            self.assert_equals(0, mcp, pv_2, gt_bot, p=p)


    @staticmethod
    def _is_signi(pval, sl):
        """Cluster have to be significant in the desired window."""
        sl = [sl] if not isinstance(sl, list) else sl
        gt = np.zeros(pval.shape, dtype=bool)
        for k in sl:
            gt[:, k] = True
        is_signi = pval < .05
        np.testing.assert_array_equal(gt, is_signi)

    def test_cluster_correction_mcp(self):
        """Test function cluster_correction_mcp."""
        n_pts, n_roi = 10, 2
        sl_neg = slice(2, 4)
        sl_pos = slice(6, 8)
        sl_both = [sl_neg, sl_pos]
        # generate the mutual information
        mi = rnd.uniform(0, 1, (n_roi, n_pts))
        mi_p = rnd.uniform(0, 1, (1000, n_roi, n_pts))
        mi[:, sl_neg] -= 1000
        mi[:, sl_pos] += 1000
        # tail = -1
        pv_neg = cluster_correction_mcp(mi, mi_p, th=-100, tail=-1)
        self._is_signi(pv_neg, sl_neg)
        # tail = 1
        pv_pos = cluster_correction_mcp(mi, mi_p, th=100, tail=1)
        self._is_signi(pv_pos, sl_pos)
        # tail = 0
        pv_both = cluster_correction_mcp(mi, mi_p, th=100, tail=0)
        self._is_signi(pv_both, sl_both)

    def test_cluster_threshold(self):
        """Test function cluster_threshold."""
        x = np.random.rand(10, 20)
        x_p = np.random.rand(100, 10, 20)

        # automatic definition
        for tfce in [False, True]:
            for tail in [-1, 0, 1]:
                th = cluster_threshold(x, x_p, tail=tail, tfce=tfce)
                if tfce:
                    assert isinstance(th, dict)
                    assert ('start' in th.keys()) and ('step' in th.keys())
                else:
                    assert isinstance(th, (int, float))
        # manual tfce definition
        th = cluster_threshold(x, x_p, tfce={'start': .5, 'step': .1})
        assert (th['start'] == .5) and (th['step'] == .1)
        assert (th['h_power'] == 2) and (th['e_power'] == .5)
        # setting tfce parameters
        th = cluster_threshold(x, x_p, tfce={'n_steps': 10, 'h_power': 1.,
                                             'e_power': .1})
        assert (th['h_power'] == 1.) and (th['e_power'] == .1)
        th = cluster_threshold(x, x_p, tfce={'e_power': .1})
        assert (th['h_power'] == 2.) and (th['e_power'] == .1)



if __name__ == '__main__':
    TestMCP().test_wise_mcp_correction()
