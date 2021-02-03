"""Test SubjectEphy and internal conversions."""
import numpy as np
import xarray as xr
import pandas as pd
import mne

from frites.dataset import SubjectEphy
from frites.utils.perf import id as id_arr


x_3d = np.random.rand(10, 5, 100)
x_4d = np.random.rand(10, 5, 4, 100)
sfreq = 64.
times = (np.arange(100) / sfreq) - .1
freqs = [4., 10., 30., 200.]
y_int = [0] * 4 + [1] * 6
y_int_2 = [0] * 2 + [1] * 4 + [2] * 4
y_flo = np.random.normal(size=(10,))
z = [2] * 5 + [3] * 5
roi = ['roi_0', 'roi_0', 'roi_1', 'roi_2', 'roi_1']
ch_names = [f"ch{k}" for k in range(len(roi))]
kw = dict(verbose='ERROR')


class TestSubjectEphy(object):  # noqa

    @staticmethod
    def _test_memory(x, y):
        """Test that no internal copy have been made."""
        assert id_arr(x) == id_arr(y)

    @staticmethod
    def _get_data(dtype, ndim):
        if (dtype == 'xr') and (ndim == 3):
            midx_tr = pd.MultiIndex.from_arrays(
                (y_int, y_flo, z), names=('y_int', 'y_flo', 'z'))
            x_out = xr.DataArray(x_3d, dims=('trials', 'roi', 'times'),
                                 coords=(midx_tr, roi, times))
        elif (dtype == 'xr') and (ndim == 4):
            midx_tr = pd.MultiIndex.from_arrays(
                (y_int, y_flo, z), names=('y_int', 'y_flo', 'z'))
            x_out = xr.DataArray(x_4d, coords=(midx_tr, roi, freqs, times),
                                 dims=('trials', 'roi', 'freqs', 'times'))
        elif (dtype == 'mne') and (ndim == 3):
            info = mne.create_info(ch_names, sfreq, ch_types='seeg')
            x_out = mne.EpochsArray(x_3d, info, tmin=times[0])
        elif (dtype == 'mne') and (ndim == 4):
            info = mne.create_info(ch_names, sfreq, ch_types='seeg')
            x_out = mne.time_frequency.EpochsTFR(info, x_4d, times, freqs)

        return x_out

    def test_numpy_inputs(self):
        """Test function numpy_inputs."""
        # ___________________________ test 3d inputs __________________________
        SubjectEphy(x_3d, **kw)
        SubjectEphy(x_3d, y=y_int, **kw)
        SubjectEphy(x_3d, z=z, **kw)
        SubjectEphy(x_3d, y=y_int, z=z, roi=roi, **kw)
        da_3d = SubjectEphy(x_3d, y=y_int, z=z, roi=roi, times=times, **kw)
        self._test_memory(x_3d, da_3d.data)

        # ___________________________ test 4d inputs __________________________
        SubjectEphy(x_4d, **kw)
        SubjectEphy(x_4d, y=y_int, **kw)
        SubjectEphy(x_4d, z=z, **kw)
        SubjectEphy(x_4d, y=y_int, z=z, roi=roi, **kw)
        da_4d = SubjectEphy(x_4d, y=y_int, z=z, roi=roi, times=times, **kw)
        self._test_memory(x_4d, da_4d.data)

    def test_xr_inputs(self):
        """Test function xr_inputs."""
        # ___________________________ test 3d inputs __________________________
        # test inputs
        xr_3d = self._get_data('xr', 3)
        SubjectEphy(xr_3d, **kw)
        SubjectEphy(xr_3d, y='y_int', **kw)
        SubjectEphy(xr_3d, y='y_flo', **kw)
        SubjectEphy(xr_3d, z='z', **kw)
        SubjectEphy(xr_3d, y='y_int', z='z', roi='roi', **kw)
        da_3d = SubjectEphy(xr_3d, y='y_flo', z='z', roi='roi', times='times',
                            **kw)
        self._test_memory(x_3d, da_3d.data)

        # ___________________________ test 4d inputs __________________________
        # test inputs
        xr_4d = self._get_data('xr', 4)
        xr_4d = self._get_data('xr', 4)
        SubjectEphy(xr_4d, **kw)
        SubjectEphy(xr_4d, y='y_int', **kw)
        SubjectEphy(xr_4d, y='y_flo', **kw)
        SubjectEphy(xr_4d, z='z', **kw)
        SubjectEphy(xr_4d, y='y_int', z='z', roi='roi', **kw)
        da_4d = SubjectEphy(xr_4d, y='y_flo', z='z', roi='roi', times='times',
                            **kw)
        self._test_memory(x_4d, da_4d.data)

    def test_mne_inputs(self):
        """Test function mne_inputs."""
        # ___________________________ test 3d inputs __________________________
        # test inputs
        mne_3d = self._get_data('mne', 3)
        SubjectEphy(mne_3d, **kw)
        SubjectEphy(mne_3d, y=y_int, **kw)
        SubjectEphy(mne_3d, z=z, **kw)
        SubjectEphy(mne_3d, y=y_int, z=z, roi=roi, **kw)
        da_3d = SubjectEphy(mne_3d, y=y_int, z=z, roi=roi, times=times, **kw)
        self._test_memory(x_3d, da_3d.data)

        # ___________________________ test 4d inputs __________________________
        # test inputs
        mne_4d = self._get_data('mne', 4)
        SubjectEphy(mne_4d, **kw)
        SubjectEphy(mne_4d, y=y_int, **kw)
        SubjectEphy(mne_4d, z=z, **kw)
        SubjectEphy(mne_4d, y=y_int, z=z, roi=roi, **kw)
        da_4d = SubjectEphy(mne_4d, y=y_int, z=z, roi=roi, times=times, **kw)
        self._test_memory(x_4d, da_4d.data)

    def test_coordinates(self):
        """Test if coordinates and dims are properly set"""
        # _________________________ Test Xarray coords ________________________
        # build the 4d data
        xrd = self._get_data('xr', 4)
        da = SubjectEphy(xrd, y='y_flo', z='z', roi='roi', times='times', **kw)
        # testings
        np.testing.assert_array_equal(y_flo, da['y'].data)
        np.testing.assert_array_equal(z, da['z'].data)
        np.testing.assert_array_equal(roi, da['roi'].data)
        np.testing.assert_array_equal(freqs, da['freqs'].data)
        np.testing.assert_array_equal(times, da['times'].data)
        np.testing.assert_array_equal(
            ('trials', 'roi', 'freqs', 'times'), da.dims)

        # ___________________________ Test MNE coords _________________________
        # build the 4d data
        mnd = self._get_data('mne', 4)
        da = SubjectEphy(mnd, y=y_flo, z=z, **kw)
        assert da.attrs['sfreq'] == sfreq
        # testings
        np.testing.assert_array_equal(y_flo, da['y'].data)
        np.testing.assert_array_equal(z, da['z'].data)
        np.testing.assert_array_equal(ch_names, da['roi'].data)
        np.testing.assert_array_equal(freqs, da['freqs'].data)
        np.testing.assert_array_equal(times, da['times'].data)
        np.testing.assert_array_equal(
            ('trials', 'roi', 'freqs', 'times'), da.dims)

    def test_multiconditions(self):
        """Test function multicond."""
        ds = SubjectEphy(x_3d, y=np.c_[y_int, y_int_2], **kw)
        np.testing.assert_array_equal(
            ds['y'].data, [0, 0, 1, 1, 2, 2, 3, 3, 3, 3])

    def test_multivariate(self):
        """Test support for multi-variate axis."""
        mnd = self._get_data('mne', 4)
        da_mv = SubjectEphy(mnd, y=y_flo, z=z, multivariate=True, **kw)
        assert da_mv.dims[-2] == 'mv'
        da_mv = SubjectEphy(mnd, y=y_flo, z=z, multivariate=False, **kw)
        assert da_mv.dims[-2] == 'freqs'

    def test_agg_ch(self):
        """Test function agg_ch."""
        xr_3d = self._get_data('xr', 3)
        da_a = SubjectEphy(xr_3d, y='y_flo', z='z', roi='roi', times='times',
                           agg_ch=True, **kw)
        # test aggregation
        np.testing.assert_array_equal(da_a['agg_ch'].data, [0] * len(roi))
        # test no aggregation
        da_na = SubjectEphy(xr_3d, y='y_flo', z='z', roi='roi', times='times',
                            agg_ch=False, **kw)
        np.testing.assert_array_equal(
            da_na['agg_ch'].data, np.arange(len(roi)))

    def test_name(self):
        """Test settings dataarray name"""
        name = 'TestingName'
        da = SubjectEphy(x_4d, name=name, **kw)
        assert da.name == name

    def test_attrs(self):
        """Test setting attributes"""
        # test attrs passed as inputs
        attrs = {'test': 'passed', 'ruggero': 'bg'}
        da = SubjectEphy(x_4d, attrs=attrs, **kw)
        assert all([da.attrs[k] == v for k, v in attrs.items()])
        # test attrs attached to an input xarray
        xr_4d = self._get_data('xr', 4)
        xr_4d.attrs = attrs
        da = SubjectEphy(xr_4d, y='y_flo', z='z', times='times', agg_ch=False,
                         multivariate=True, **kw)
        assert all([da.attrs[k] == v for k, v in attrs.items()])
        # test computed attrs
        assert da.attrs['sfreq'] == sfreq
        assert da.attrs['y_dtype'] == 'float'
        assert da.attrs['z_dtype'] == 'int'
        assert da.attrs['mi_type'] == 'ccd'
        assert da.attrs['agg_ch'] is False
        assert da.attrs['multivariate'] is True

    def test_dtypes(self):
        """Test y, z dtypes and mi_type."""
        # cd
        da = SubjectEphy(x_3d, y=y_int, **kw)
        assert da.attrs['y_dtype'] == 'int'
        assert da.attrs['z_dtype'] == 'none'
        assert da.attrs['mi_type'] == 'cd'
        # cc
        da = SubjectEphy(x_3d, y=y_flo, **kw)
        assert da.attrs['y_dtype'] == 'float'
        assert da.attrs['z_dtype'] == 'none'
        assert da.attrs['mi_type'] == 'cc'
        # ccd
        da = SubjectEphy(x_3d, y=y_flo, z=z, **kw)
        assert da.attrs['y_dtype'] == 'float'
        assert da.attrs['z_dtype'] == 'int'
        assert da.attrs['mi_type'] == 'ccd'


if __name__ == '__main__':
    TestSubjectEphy().test_dtypes()
