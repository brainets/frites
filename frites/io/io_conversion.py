"""Set of functions for converting outputs."""
import numpy as np

from .io_dependencies import is_pandas_installed, is_xarray_installed


def convert_spatiotemporal_outputs(arr, index=None, columns=None,
                                   astype='array'):
    """Convert spatio-temporal outputs.

    Parameters
    ----------
    arr : array_like
        2d array of shape (n_times, n_roi)
    index : array_like | None
        Array of roi names
    columns : array_like | None
        Array of time index
    astype : {'array', 'dataframe', 'dataarray'}
        Convert the array either to a pandas DataFrames (require pandas to be
        installed) either to a xarray DataArray (require xarray to be
        installed)

    Returns
    -------
    arr_c : array_like | DataFrame | DataArray
        Converted spatio-temporal array
    """
    assert isinstance(arr, np.ndarray) and (arr.ndim == 2)
    assert astype in ['array', 'dataframe', 'dataarray']
    # checkout index and columns
    if index is None:
        index = np.arange(arr.shape[0])
    if columns is None:
        columns = np.arange(arr.shape[1])
    assert arr.shape == (len(index), len(columns))
    # output conversion
    force_np = not is_pandas_installed() and not is_xarray_installed()
    if (astype is 'array') or force_np:       # numpy
        return arr
    elif astype is 'dataframe':               # pandas
        is_pandas_installed(raise_error=True)
        import pandas as pd
        return pd.DataFrame(arr, index=index, columns=columns)
    elif astype is 'dataarray':               # xarray
        is_xarray_installed(raise_error=True)
        from xarray import DataArray
        return DataArray(arr, dims=('times', 'roi'), coords=(index, columns))
