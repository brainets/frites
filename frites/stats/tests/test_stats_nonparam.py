"""Test high-level mutual information functions."""
import numpy as np
import xarray as xr

from frites.stats import (permute_mi_vector, permute_mi_trials,
                          bootstrap_partitions, dist_to_ci,
                          confidence_interval)

rnd = np.random.RandomState(0)

mi_types = ['cc', 'cd', 'ccd']
inferences = ['ffx', 'rfx']
n_perm = 15

n_times, n_epochs, n_suj, n_conds = 100, 100, 2, 3
y = rnd.rand(n_epochs)
suj = np.round(np.linspace(0, n_suj, n_epochs)).astype(int)


class TestNonParam(object):  # noqa

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

    def test_bootstrap_partitions(self):
        """Test function bootstrap_partitions."""
        # overall testing
        part = bootstrap_partitions(10, n_partitions=5)
        assert len(part) == 5
        for k in part:
            assert (0 <= k).all() and (k < 10).all()

        # group testing
        gp_1 = np.array([0, 0, 0, 1, 1, 1])
        part = bootstrap_partitions(6, gp_1, n_partitions=5)
        for k in part:
            assert (0 <= k[0:3].min()) and (k[0:3].max() <= 2)
            assert (3 <= k[3::].min()) and (k[3::].max() <= 5)

        gp_1 = np.array([0, 0, 1, 1, 2, 2])
        part = bootstrap_partitions(6, gp_1, n_partitions=5)
        for k in part:
            assert (0 <= k[0:2].min()) and (k[0:2].max() <= 1)
            assert (2 <= k[2:4].min()) and (k[2:4].max() <= 3)
            assert (4 <= k[4::].min()) and (k[4::].max() <= 5)

    def test_dist_to_ci(self):
        """Test function dist_to_ci."""
        # standard definition
        dist = np.random.rand(200, 1, 30)
        assert dist_to_ci(dist, cis=[95]).shape == (1, 2, 30)
        assert dist_to_ci(dist, cis=[95, 99, 99.9]).shape == (3, 2, 30)

        # mean effect size
        dist = np.random.rand(200, 10, 30)
        assert dist_to_ci(dist, cis=[95], inference='rfx').shape == (1, 2, 30)
        assert dist_to_ci(
            dist, cis=[95, 99, 99.9], inference='rfx').shape == (3, 2, 30)

    def test_confidence_interval(self):
        """Test function confidence_interval."""
        # parameters
        ci_a = [90, 90.5, 'sd', 'sem']

        # define a dataset
        dist = np.random.rand(10, 100, 30)
        coords = [np.arange(k) for k in dist.shape]
        dims = [str(k) for k in range(dist.ndim)]
        dist_xr = xr.DataArray(
            dist, dims=dims, coords=coords, attrs={'arg': 'test'}, name='Test'
        )

        for d in [dist, dist_xr]:
            # shape testing
            ci = confidence_interval(d, axis=0)
            assert ci.shape == (1, 2, 100, 30)
            ci = confidence_interval(d, axis=1)
            assert ci.shape == (1, 2, 10, 30)
            ci = confidence_interval(d, axis=2)
            assert ci.shape == (1, 2, 10, 100)
            if isinstance(d, xr.DataArray):
                ci = confidence_interval(d, axis='2')
                assert ci.shape == (1, 2, 10, 100)

            # testing individual supported cis
            for cis in ci_a:
                ci = confidence_interval(d, axis=2, cis=cis)
                assert ci.shape == (1, 2, 10, 100)

            # test multiple ci at once
            ci = confidence_interval(d, axis=2, cis=ci_a)
            assert ci.shape == (4, 2, 10, 100)

            # test random state
            ci_1 = confidence_interval(d, axis=2, cis=ci_a, random_state=0)
            ci_2 = confidence_interval(d, axis=2, cis=ci_a, random_state=0)
            ci_3 = confidence_interval(d, axis=2, cis=ci_a, random_state=1)
            np.testing.assert_array_equal(ci_1, ci_2)
            assert ~np.all(ci_1 == ci_3)


if __name__ == '__main__':
    TestNonParam().test_confidence_interval()
