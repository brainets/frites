"""Test the statistical workflow of electrophysiological data."""
import numpy as np


from frites.stats import STAT_FUN
from frites.workflow import WfStatsEphy

n_roi = 5
n_times = 20
n_perm = 200
n_suj_rfx = 15
sl = slice(5, 15)
effect_size = 100.


class TestWfStatsEphy(object):
    """Test the WfStatsEphy workflow."""

    @staticmethod
    def _generate_ffx_data(random_state):
        """Generate ffx random data."""
        rnd = np.random.RandomState(random_state)
        effect = [rnd.rand(1, n_times) for k in range(n_roi)]
        for k in range(n_roi):
            effect[k][0, sl] *= effect_size
        perms = [rnd.rand(n_perm, 1, n_times) for k in range(n_roi)]

        return effect, perms

    @staticmethod
    def _generate_rfx_data(random_state):
        """Generate rfx random data."""
        rnd = np.random.RandomState(random_state)
        effect = [rnd.rand(n_suj_rfx, n_times) for k in range(n_roi)]
        for k in range(n_roi):
            effect[k][:, sl] *= effect_size
        perms = [rnd.rand(n_perm, n_suj_rfx, n_times) for k in range(n_roi)]

        return effect, perms

    @staticmethod
    def _test_outputs(pv, tv, inference, stat):
        """Test outputs significance."""
        if inference == 'ffx':
            assert tv is None
        else:
            assert tv.shape == (n_times, n_roi)
        assert pv.shape == (n_times, n_roi)
        # ground truth construction
        ground_truth = np.zeros((n_times, n_roi), dtype=int)
        ground_truth[sl, :] = 1
        # test number of elements that are equals
        is_signi = (pv < .05).astype(int)
        p = 100. * (is_signi == ground_truth).sum() / (n_times * n_roi)
        assert (p > 95.), (f"Method={stat}; Prob={p}")

    def test_definition(self):
        """Test the definition of a workflow."""
        TestWfStatsEphy()

    def test_fit(self):
        """Test fitting the worflow."""
        wf = WfStatsEphy(verbose=False)

        # test none
        effect, perms = self._generate_ffx_data(0)
        pv, tv = wf.fit(effect, perms, stat_method=None)
        pv_def = np.ones((n_times, n_roi), dtype=float)
        np.testing.assert_array_equal(pv, pv_def)
        assert tv is None
        # test ffx
        for stat in STAT_FUN['ffx'].keys():
            pv, tv = wf.fit(effect, perms, stat_method=stat)
            self._test_outputs(pv, tv, 'ffx', stat)
        # test rfx
        effect, perms = self._generate_rfx_data(0)
        for stat in STAT_FUN['rfx'].keys():
            pv, tv = wf.fit(effect, perms, stat_method=stat)
            self._test_outputs(pv, tv, 'rfx', stat)
