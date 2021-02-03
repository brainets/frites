"""Test definition of electrophysiological datasets."""
import numpy as np
import xarray as xr
import pandas as pd

from frites.dataset import DatasetEphy
from frites.simulations import sim_multi_suj_ephy, sim_mi_cc
from frites.core import copnorm_cat_nd, copnorm_nd

np.random.seed(0)


# NumPy defintion
n_subjects = 2
x_3d = [np.random.rand(10, 5, 100) + 10. for n_s in range(n_subjects)]
x_4d = [np.random.rand(10, 5, 4, 100) + 10. for n_s in range(n_subjects)]
sfreq = 64.
times = (np.arange(100) / sfreq) - .1
freqs = [4., 10., 30., 200.]
y_int = [[0] * 4 + [1] * 6, [0] * 2 + [1] * 4 + [2] * 4]
y_flo = [np.random.normal(size=(10,)) for n_s in range(n_subjects)]
z = [[2] * 5 + [3] * 5, [3] * 5 + [2] * 5]
roi = [
    ['roi_0', 'roi_0', 'roi_1', 'roi_2', 'roi_1'],
    ['roi_1', 'roi_0', 'roi_1', 'roi_2', 'roi_3']
]

kw = dict(verbose='ERROR')


class TestDatasetEphy(object):  # noqa

    @staticmethod
    def _get_data(ndim):
        # multi-indexing along trials dimension
        midx = [pd.MultiIndex.from_arrays(
            [k, i], names=['y', 'z']) for k, i in zip(y_int, z)]
        if ndim == 3:
            x_out = [xr.DataArray(
                x_3d[k], dims=('trials', 'roi', 'times'),
                coords=(midx[k], roi[k], times)) for k in range(n_subjects)]
        elif ndim == 4:
            x_out = [xr.DataArray(
                x_4d[k], dims=('trials', 'roi', 'freqs', 'times'),
                coords=(midx[k], roi[k], freqs,
                        times)) for k in range(n_subjects)]

        return x_out

    def test_definition(self):
        """Test function definition."""
        d_3d = self._get_data(3)
        DatasetEphy(d_3d.copy(), **kw)
        DatasetEphy(d_3d.copy(), y='y', **kw)
        DatasetEphy(d_3d.copy(), y='y', z='z', **kw)
        DatasetEphy(d_3d.copy(), y='y', z='z', roi='roi', **kw)
        DatasetEphy(d_3d.copy(), y='y', z='z', roi='roi', times='times', **kw)
        DatasetEphy(d_3d.copy(), y='y', z='z', roi='roi', times='times',
                    agg_ch=False, **kw)
        DatasetEphy(d_3d, y='y', z='z', roi='roi', times='times', agg_ch=False,
                    multivariate=True, **kw)

    def test_multiconditions(self):
        """Test multi-conditions remapping."""
        d_3d = self._get_data(3)
        ds = DatasetEphy(d_3d, y=[np.c_[k, i] for k, i in zip(y_int, z)], **kw)
        y_s1, y_s2 = ds.x[0]['y'].data, ds.x[1]['y'].data
        np.testing.assert_array_equal(y_s1, [0] * 4 + [1] + [2] * 5)
        np.testing.assert_array_equal(y_s2, [3] * 2 + [2] * 3 + [1] + [4] * 4)

    def test_get_roi_data(self):
        """Test getting the data of a single brain region."""
        # build dataset
        d_3d = self._get_data(3)
        ds = DatasetEphy(d_3d, y='y', z='z', **kw)
        # get the data
        ds_roi2 = ds.get_roi_data("roi_2", copnorm=False)
        np.testing.assert_array_equal(ds_roi2.shape, (100, 1, 20))
        # test the data
        s1_r2, s2_r2 = d_3d[0].sel(roi='roi_2'), d_3d[1].sel(roi='roi_2')
        s12 = xr.concat((s1_r2, s2_r2), 'trials').T.expand_dims('mv', axis=-2)
        np.testing.assert_array_equal(ds_roi2.data, s12.data)
        # test task-related variables
        y_12, z_12 = np.r_[y_int[0], y_int[1]], np.r_[z[0], z[1]]
        np.testing.assert_array_equal(y_12, ds_roi2['y'].data)
        np.testing.assert_array_equal(z_12, ds_roi2['z'].data)

    def test_agg_ch(self):
        """Test channels aggregation."""
        # build dataset (with aggregation)
        d_3d = self._get_data(3)
        ds = DatasetEphy(d_3d, roi='roi', agg_ch=True, **kw)
        ds_roi2 = ds.get_roi_data("roi_0", copnorm=False)
        np.testing.assert_array_equal(ds_roi2['agg_ch'].data, [0] * 30)
        # build dataset (without aggregation)
        ds = DatasetEphy(d_3d, roi='roi', agg_ch=False, **kw)
        ds_roi0 = ds.get_roi_data("roi_0", copnorm=False)
        np.testing.assert_array_equal(
            ds_roi0['agg_ch'].data, [0] * 10 + [1] * 10 + [6] * 10)

    def test_nb_min_suj(self):
        """Test if the selection based on a minimum number of subjects."""
        d_3d = self._get_data(3)
        # nb_min_suj = -inf
        ds = DatasetEphy(d_3d, roi='roi', nb_min_suj=None, **kw)
        assert len(ds.roi_names) == 4
        conn_ud = np.c_[ds.get_connectivity_pairs(directed=False)]
        assert conn_ud.shape == (6, 2)
        conn_d = np.c_[ds.get_connectivity_pairs(directed=True)]
        assert conn_d.shape == (12, 2)
        # nb_min_suj = 2
        ds = DatasetEphy(d_3d, roi='roi', nb_min_suj=2, **kw)
        assert len(ds.roi_names) == 3
        np.testing.assert_array_equal(
            ds.roi_names, ['roi_0', 'roi_1', 'roi_2'])
        conn_ud = np.c_[ds.get_connectivity_pairs(directed=False)]
        assert conn_ud.shape == (3, 2)
        conn_d = np.c_[ds.get_connectivity_pairs(directed=True)]
        assert conn_d.shape == (6, 2)

    def test_copnorm(self):
        """Test function copnorm."""
        # build dataset
        d_3d = self._get_data(3)
        ds = DatasetEphy(d_3d, y='y', z='z', **kw)
        # check copnorm range
        ds_roi2 = ds.get_roi_data("roi_2", copnorm=False)
        s1_r2, s2_r2 = d_3d[0].sel(roi='roi_2'), d_3d[1].sel(roi='roi_2')
        s12 = xr.concat((s1_r2, s2_r2), 'trials').T.expand_dims('mv', axis=-2)
        assert 9. < ds_roi2.data.ravel().mean() < 11.
        np.testing.assert_array_equal(s12.data, ds_roi2.data)
        ds_roi2 = ds.get_roi_data("roi_2", copnorm=True)
        assert -1. < ds_roi2.data.ravel().mean() < 1.
        # check values (gcrn_per_suj=False)
        gc_t = ds.get_roi_data("roi_2", copnorm=True, gcrn_per_suj=False)
        np.testing.assert_array_equal(copnorm_nd(s12.data), gc_t.data)
        # check values (gcrn_per_suj=True)
        gc_t = ds.get_roi_data("roi_2", copnorm=True, gcrn_per_suj=True)
        np.testing.assert_array_equal(
            copnorm_cat_nd(s12.data, gc_t['subject'].data), gc_t.data)

    def test_multivariate(self):
        """Test multivariate"""
        d_4d = self._get_data(4)
        # multivariate = False
        ds = DatasetEphy(d_4d, roi='roi', multivariate=False, **kw)
        x_roi2 = ds.get_roi_data('roi_2')
        assert x_roi2.dims == ('freqs', 'times', 'mv', 'rtr')
        assert x_roi2.shape == (4, 100, 1, 20)
        # multivariate = True
        d_4d = self._get_data(4)
        ds = DatasetEphy(d_4d, roi='roi', multivariate=True, **kw)
        x_roi2 = ds.get_roi_data('roi_2')
        assert x_roi2.dims == ('times', 'mv', 'rtr')
        assert x_roi2.shape == (100, 4, 20)

    def test_properties(self):
        """Test function properties."""
        d_3d = self._get_data(3)
        ds = DatasetEphy(d_3d, y='y', z='z', roi='roi', times='times', **kw)
        assert isinstance(ds.x, list)
        assert isinstance(ds.df_rs, pd.DataFrame)
        np.testing.assert_array_equal(ds.times, times)
        np.testing.assert_array_equal(
            ds.roi_names, ['roi_0', 'roi_1', 'roi_2', 'roi_3'])

    def test_builtin(self):
        """Test function builtin."""
        d_3d = self._get_data(3)
        ds = DatasetEphy(d_3d, y='y', z='z', **kw)
        print(ds)

    def test_slicing(self):
        """Test spatio-temporal slicing."""
        d_3d = self._get_data(3)
        xt1, xt2 = 0.1, 0.5
        xs1, xs2 = np.abs(times - xt1).argmin(), np.abs(times - xt2).argmin()
        # ds.sel
        ds = DatasetEphy(d_3d, times='times', **kw)
        ds = ds.sel(times=slice(xt1, xt2))
        np.testing.assert_array_equal(ds.times, times[slice(xs1, xs2 + 1)])
        # ds.isel
        ds = DatasetEphy(d_3d, times='times', **kw)
        ds = ds.isel(times=slice(xs1, xs2))
        np.testing.assert_array_equal(ds.times, times[slice(xs1, xs2)])

    def test_savgol_filter(self):
        """Test function savgol_filter."""
        d_3d = self._get_data(3)
        ds = DatasetEphy(d_3d, times='times', **kw)
        ds.savgol_filter(10., verbose=False)


if __name__ == '__main__':
    TestDatasetEphy().test_definition()
