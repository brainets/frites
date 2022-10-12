"""Test parametric functions."""
import numpy as np

from frites.stats.stats_param import (ttest_1samp, rfx_ttest)


class TestParam(object):  # noqa

    def test_ttest_1samp(self):
        """Test function ttest_1samp."""
        x = np.random.rand(2, 100, 3)
        popmean = 0.
        tt_sci = ttest_1samp(x, popmean, axis=1, implementation='scipy')
        tt_mne = ttest_1samp(x, popmean, axis=1, implementation='mne')
        assert tt_sci.shape == tt_mne.shape == (2, 3)

    def test_rfx_ttest(self):
        """Test function rfx_ttest."""
        n_suj, n_roi, n_times, n_perm = 4, 5, 10, 50
        x = [np.random.rand(n_suj, n_times) for _ in range(n_roi)]
        x_p = [np.random.rand(n_perm, n_suj, n_times) for _ in range(n_roi)]
        # basic version
        tv, tv_p, _ = rfx_ttest(x, x_p)
        assert tv.shape == (n_roi, n_times)
        assert tv_p.shape == (n_perm, n_roi, n_times)
        # center
        for center in [False, True, 'mean', 'median', 'trimmed', 'zscore']:
            tv, tv_p, _ = rfx_ttest(x, x_p, center=center)
            assert tv.shape == (n_roi, n_times)
            assert tv_p.shape == (n_perm, n_roi, n_times)
        # t-tested
        tv, tv_p, _ = rfx_ttest(x, x_p, ttested=True)
        assert tv.shape == (n_roi * n_suj, n_times)
        assert tv_p.shape == (n_perm, n_roi * n_suj, n_times)
