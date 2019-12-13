"""Test the statistical workflow of electrophysiological data."""
import numpy as np
from itertools import product

from frites.workflow import WfStatsEphy


n_roi = 5
n_times = 20
n_perm = 30
n_suj = 10
cl_pos, cl_neg = slice(5, 10), slice(12, 17)

# full RFX dataset definition
effect_rfx, perms_rfx = [], []
for k in range(n_roi):
    rnd = np.random.RandomState(k)
    _x = rnd.rand(n_suj, n_times)
    _x[:, cl_pos] += 1000.
    _x[:, cl_neg] -= 1000.
    _x_p = rnd.rand(n_perm, n_suj, n_times)
    effect_rfx += [_x]
    perms_rfx += [_x_p]
# full FFX dataset definition
effect_ffx = [k[[0], :] for k in effect_rfx]
perms_ffx = [k[:, [0], :] for k in perms_rfx]
# ground truth definition
gt_pos = np.zeros((n_roi, n_times), dtype=int)
gt_pos[:, cl_pos] = 1
gt_neg = np.zeros((n_roi, n_times), dtype=int)
gt_neg[:, cl_neg] = 1
gt_bot = np.zeros((n_roi, n_times), dtype=int)
gt_bot[:, cl_pos] = 1
gt_bot[:, cl_neg] = 1


class TestWfStatsEphy(object):  # noqa

    @staticmethod
    def _testing(gt, pv, settings, alpha=0.05, tolerance=0.05):
        p_correct = 1. - ((pv < alpha).astype(int) == gt).sum() / gt.size
        assert p_correct < tolerance, f"{settings}"

    def test_fit(self):
        """Test running the workflow."""
        # ---------------------------------------------------------------------
        # loop parameters
        inferences = ['ffx', 'rfx']
        mcps = ['maxstat', 'fdr', 'bonferroni']
        levels = ['testwise', 'cluster']
        prod = product(inferences, mcps, levels)

        # definition of the workflow
        wf = WfStatsEphy(verbose=False)

        for inf, mcp, level in prod:
            # data selection
            if inf is 'ffx': x, x_p = effect_ffx, perms_ffx  # noqa
            elif inf is 'rfx': x, x_p = effect_rfx, perms_rfx  # noqa
            # threshold definition
            if level is 'testwise':
                cluster_th = [None]
            if level is 'cluster':
                cluster_th = [None, 'tfce']

            for th in cluster_th:
                # upper tail
                kw = dict(mcp=mcp, level=level, inference=inf, tail=1,
                          cluster_th=th)
                pv, tv = wf.fit(x, x_p, **kw)
                self._testing(gt_pos.T, pv, kw)
                # lower tail
                kw = dict(mcp=mcp, level=level, inference=inf, tail=-1,
                          cluster_th=th)
                pv, tv = wf.fit(x, x_p, **kw)
                self._testing(gt_neg.T, pv, kw)
                # both tails
                kw = dict(mcp=mcp, level=level, inference=inf, tail=0,
                          cluster_th=th)
                pv, tv = wf.fit(x, x_p, **kw)
                self._testing(gt_bot.T, pv, kw)
