"""Test simulating distant MI."""
import numpy as np

from frites.simulations import (sim_gauss_fit, sim_distant_cc_ms,
                                sim_distant_cc_ss)


class TestDistantMI(object):

    def test_sim_gauss_fit(self):
        for stim_type in ['discrete_stim', 'cont_linear', 'cont_flat']:
            x, y, stim = sim_gauss_fit(stim_type=stim_type, n_epochs=20)
            assert x.shape == y.shape
            if stim_type is not 'discrete_stim':
                assert len(stim) == x.shape[0]

    def test_sim_distant_cc_ss(self):
        n_epochs = 20
        x, y, roi = sim_distant_cc_ss(n_epochs=n_epochs)
        assert x.shape == (len(y), len(roi), 400)

    def test_sim_distant_cc_ms(self):
        n_subjects = 3
        x, y, roi, times = sim_distant_cc_ms(n_subjects)
        assert len(x) == len(y) == len(roi) == n_subjects
        assert [k.shape == x[0].shape for k in x]
        assert [k.shape == y[0].shape for k in y]
        assert [k.shape == roi[0].shape for k in roi]
        assert x[0].shape == (len(y[0]), len(roi[0]), len(times))
