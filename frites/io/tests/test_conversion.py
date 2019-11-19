"""Test functions that convert."""
import numpy as np

from frites.io import convert_spatiotemporal_outputs


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
        n_times, n_roi = 10, 5
        roi = np.array([f'roi_{k}' for k in range(n_roi)])
        times = np.arange(n_times)
        arr = np.random.rand(n_times, n_roi)
        # NumPy
        arr_def = convert_spatiotemporal_outputs(arr)
        self._test_type(arr_def, 'array')
        arr_dim = convert_spatiotemporal_outputs(arr, times, roi)
        self._test_type(arr_dim, 'array')
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


if __name__ == '__main__':
    TestIOConversion().test_convert_spatiotemporal_outputs()
