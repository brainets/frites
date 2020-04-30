"""Set of functions for converting outputs."""
import numpy as np

from frites.io.io_dependencies import is_pandas_installed, is_xarray_installed


def convert_spatiotemporal_outputs(arr, times, roi, astype='array'):
    """Convert spatio-temporal outputs.

    Parameters
    ----------
    arr : array_like
        2d array of shape (n_times, n_roi)
    times : array_like | None
        Array of roi names
    roi : array_like | None
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
    assert arr.shape == (len(times), len(roi))
    # output conversion
    force_np = not is_pandas_installed() and not is_xarray_installed()
    astype = 'array' if force_np else astype
    if astype is 'array':                     # numpy
        return arr
    elif astype is 'dataframe':               # pandas
        is_pandas_installed(raise_error=True)
        import pandas as pd
        return pd.DataFrame(arr, index=times, columns=roi)
    elif astype is 'dataarray':               # xarray
        is_xarray_installed(raise_error=True)
        from xarray import DataArray
        return DataArray(arr, dims=('times', 'roi'), coords=(times, roi))


def convert_dfc_outputs(arr, times, roi, sources, targets, astype='2d_array',
                        is_pvalue=False):
    """Convert dynamic functional connectivity outputs.

    This functions can be used to convert an array of dynamical functional
    connectivity (dFC) from a shape (n_times, n_pairs) into either the same
    shape but using pandas DataFrame or to an array of shape
    (n_sources, n_targets, n_times). The number of pairs n_pairs is defined as
    the length of `sources` or `targets` inputs
    (pairs = np.c_[sources, targets]).

    Parameters
    ----------
    arr : array_like
        Array of connectivity of shape (n_times, n_pairs)
    times : array_like
        Array of time points of shape (n_times,)
    roi : array_like
        Array of region of interest names of shape (n_roi,)
    sources : array_like
        Array of sources indices of shape (n_pairs,)
    targets : array_like
        Array of targets indices of shape (n_pairs,)
    astype : {2d_array, 3d_array, 2d_dataframe, 3d_dataframe, dataarray}
        String describing the output type. Use either :

            * '2d_array', '3d_array' : NumPy arrays respectively of shapes
              (n_pairs, n_times) or (n_sources, n_targets, n_times)
            * '2d_dataframe', '3d_dataframe' : Pandas DataFrame both of shapes
              (n_pairs, n_times) but the 2d version is a single column level
              (roi_source, roi_target) while the 3d version is a muli-level
              index DataFrame. Require pandas to be installed
            * 'dataarray' : a 3d xarray DataArray of shape
              (n_sources, n_targets, n_times). Requires xarray to be installed
              but this the recommended output as slicing is much easier.
    is_pvalue : bool | False
        Specify if the array is p-values

    Returns
    -------
    arr_c : array_like | DataFrame | DataArray
        Converted dFC array
    """
    assert isinstance(arr, np.ndarray) and (arr.ndim == 2)
    assert len(sources) == len(targets)
    assert arr.shape == (len(times), len(sources))
    assert astype in ['2d_array', '3d_array', '2d_dataframe', '3d_dataframe',
                      'dataarray']
    # empty fcn to use
    empty_fcn = np.zeros if not is_pvalue else np.ones
    # get used roi and unique sources / targets
    roi = np.asarray(roi)
    s_roi, t_roi = roi[sources], roi[targets]
    n_times = arr.shape[0]
    _, s_idx = np.unique(sources, return_index=True)
    _, t_idx = np.unique(targets, return_index=True)

    # output conversion
    force_np = not is_pandas_installed() and not is_xarray_installed()
    astype = '2d_array' if force_np else astype

    if astype is '2d_array':
        return arr
    elif astype is '3d_array':
        out = empty_fcn((len(roi), len(roi), n_times))
        out[sources, targets, :] = arr.T
        return out
    elif astype is '2d_dataframe':
        import pandas as pd
        columns = [(s, t) for s, t in zip(s_roi, t_roi)]
        return pd.DataFrame(arr, index=times, columns=columns)
    elif astype is '3d_dataframe':
        import pandas as pd
        idx = pd.MultiIndex.from_arrays([s_roi, t_roi],
                                        names=['source', 'target'])
        return pd.DataFrame(arr, index=times, columns=idx)
    elif astype is 'dataarray':
        from xarray import DataArray
        out = empty_fcn((len(roi), len(roi), n_times))
        out[sources, targets, :] = arr.T
        da = DataArray(out, dims=('source', 'target', 'times'),
                       coords=(roi, roi, times))
        return da
