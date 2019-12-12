"""Test correction of Multiple Comparison Problem (MCP)."""
import numpy as np

from frites.stats import permutation_mcp_correction


class TestMCP(object):

    @staticmethod
    def assert_equals(tail, mcp, pv, gt, p=.05, tolerance=.05):
        is_signi = (pv < p).astype(int)
        p_correct = 1. - (is_signi == gt.astype(int)).sum() / gt.size
        assert p_correct < tolerance, (
            f"{mcp} - {tail} tail : {p_correct} >= tolerance={tolerance}")

    def test_permutation_mcp_correction(self):
        """Test function permutation_mcp_correction."""
        # generate som random data
        rnd = np.random.RandomState(0)
        x = rnd.rand(8, 20)
        x_p = rnd.rand(8, 20, 10000)
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
            pv_1 = permutation_mcp_correction(x, x_p, tail=1, mcp=mcp,
                                              alpha=.05, inplace=False)
            self.assert_equals(1, mcp, pv_1, gt_pos, p=p)
            # negative tail
            pv_m1 = permutation_mcp_correction(x, x_p, tail=-1, mcp=mcp,
                                               alpha=.05, inplace=False)
            self.assert_equals(-1, mcp, pv_m1, gt_neg, p=p)
            # both tails
            pv_2 = permutation_mcp_correction(x, x_p, tail=0, mcp=mcp,
                                              alpha=.05, inplace=False)
            self.assert_equals(0, mcp, pv_2, gt_bot, p=p)
