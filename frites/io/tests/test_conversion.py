"""Test functions that convert."""
import numpy as np

from frites.io import (convert_spatiotemporal_outputs, convert_dfc_outputs)


class TestIOConversion(object):

    @staticmethod
    def _test_type(arr, astype):
        if astype == 'array':
            assert isinstance(arr, np.ndarray)
        elif astype == 'dataframe':
            import pandas as pd
            assert isinstance(arr, pd.DataFrame)
        elif astype == 'dataarray':
            from xarray import DataArray
            assert isinstance(arr, DataArray)

    def test_convert_spatiotemporal_outputs(self):
        """Test function convert_spatiotemporal_outputs."""
        # ---------------------------------------------------------------------
        # tested array definition
        n_times, n_roi = 10, 5
        roi = np.array([f'roi_{k}' for k in range(n_roi)])
        times = np.arange(n_times)
        arr = np.random.rand(n_times, n_roi)

        # ---------------------------------------------------------------------
        # test independant definitions

        # NumPy
        arr_np = convert_spatiotemporal_outputs(arr, times, roi)
        self._test_type(arr_np, 'array')
        # Pandas
        arr_df = convert_spatiotemporal_outputs(arr, times, roi, 'dataframe')
        self._test_type(arr_df, 'dataframe')
        np.testing.assert_equal(arr_df.index, times)
        np.testing.assert_equal(np.array(arr_df.columns), roi)
        # Xarray
        arr_da = convert_spatiotemporal_outputs(arr, times, roi, 'dataarray')
        self._test_type(arr_da, 'dataarray')
        np.testing.assert_equal(arr_da.times, times)
        np.testing.assert_equal(arr_da.roi, roi)

        # ---------------------------------------------------------------------
        # test that all outputs are equals

        np.testing.assert_equal(arr, arr_np)
        np.testing.assert_equal(arr, np.array(arr_df))
        np.testing.assert_equal(arr, np.array(arr_da))

    def test_convert_dfc_outputs(self):
        """Test function convert_dfc_outputs."""
        # ---------------------------------------------------------------------
        # tested array definition
        n_roi, n_times = 4, 10
        sources, targets = np.where(~np.eye(n_roi, dtype=bool))
        n_pairs = len(sources)
        roi = np.array([f"roi_{k}" for k in range(n_roi)])
        times = np.linspace(-1, 1, n_times)
        # make it worst by shuffling both of them
        pairs = np.c_[sources, targets]
        np.random.shuffle(pairs)
        sources, targets = pairs.T
        # array of dFC
        arr = np.random.rand(n_times, n_pairs)
        # intentional loop definition of the 3d array
        arr_3d = np.zeros((n_roi, n_roi, n_times))
        for n, (s, t) in enumerate(zip(sources, targets)):
            arr_3d[s, t, :] = arr[:, n]
        args = (arr, times, roi, sources, targets)

        # ---------------------------------------------------------------------
        # test independant definitions

        # 2d NumPy array
        np_2d = convert_dfc_outputs(*args, astype='2d_array')
        self._test_type(np_2d, 'array')
        # 3d NumPy array
        np_3d = convert_dfc_outputs(*args, astype='3d_array')
        self._test_type(np_3d, 'array')
        # 2d Pandas DataFrame
        df_2d = convert_dfc_outputs(*args, astype='2d_dataframe')
        self._test_type(df_2d, 'dataframe')
        col_2d = np.array([(roi[s], roi[t]) for s, t in zip(sources, targets)])
        np.testing.assert_equal(df_2d.index, times)
        np.testing.assert_equal(np.array(list(df_2d.columns)), col_2d)
        # 3d Pandas DataFrame
        df_3d = convert_dfc_outputs(*args, astype='3d_dataframe')
        self._test_type(df_3d, 'dataframe')
        idx = df_3d.columns
        df_3d_sources = np.array(idx.get_level_values(0))
        df_3d_targets = np.array(idx.get_level_values(1))
        np.testing.assert_equal(df_3d.index, times)
        np.testing.assert_equal(df_3d_sources, roi[sources])
        np.testing.assert_equal(df_3d_targets, roi[targets])
        # DataArray
        da = convert_dfc_outputs(*args, astype='dataarray')
        self._test_type(da, 'dataarray')
        np.testing.assert_equal(da.times, times)
        np.testing.assert_equal(da.source, roi)
        np.testing.assert_equal(da.target, roi)

        # ---------------------------------------------------------------------
        # test that all outputs are equals

        # 2d NumPy array
        np.testing.assert_equal(arr, np_2d)
        # 3d NumPy array
        np.testing.assert_equal(arr_3d, np_3d)
        # 2d Pandas DataFrame
        np.testing.assert_equal(arr, np.array(df_2d))
        # 3d Pandas DataFrame
        df_3d_arr = np.zeros((n_roi, n_roi, n_times))
        for s, t in zip(sources, targets):
            df_3d_arr[s, t, :] = df_3d.loc[:, (f'roi_{s}', f'roi_{t}')]
        np.testing.assert_equal(arr_3d, df_3d_arr)
        # DataArray
        np.testing.assert_equal(arr_3d, np.array(da))


if __name__ == '__main__':
    TestIOConversion().test_convert_dfc_outputs()
