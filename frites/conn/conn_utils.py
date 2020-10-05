"""Utility functions for connectivity."""
import numpy as np
import xarray as xr


def conn_reshape_undirected(da, sep='-', order=None, fill_value=np.nan,
                            to_dataframe=False):
    """Reshape a raveled undirected array of connectivity.

    This function takes a DataArray of shape (n_pairs,) or (n_pairs, n_times)
    where n_pairs reflects pairs of roi (e.g 'roi_1-roi_2') and reshape it to
    be a symmetric DataArray of shape (n_roi, n_roi, n_times).

    .. warning::

        This function reshape the data inplace. If you need to conserve the
        ravel version of your connectivity array, send a copy (`da.copy()`)

    Parameters
    ----------
    da : xarray.DataArray
        Xarray DataArray of shape (n_pairs, n_times) where actually the roi
        dimension contains the pairs (roi_1-roi_2, roi_1-roi_3 etc.)
    sep : string | '-'
        Separator used to separate the pairs of roi names.
    order : list | None
        List of roi names to reorder the output.
    fill_value : float | np.nan
        Value to use for filling missing pairs (e.g diagonal)
    to_dataframe : bool | False
        Dataframe conversion. Only possible if the da input does not contains
        a time axis.

    Returns
    -------
    da_out : xarray.DataArray
        DataArray of shape (n_roi, n_roi, n_times)

    See also
    --------
    conn_dfc
    """
    assert isinstance(da, xr.DataArray)
    assert 'roi' in list(da.dims)
    if 'times' not in list(da.dims):
        da = da.expand_dims("times")

    # get sources, targets names and sorted full list
    sources, targets, roi_tot = _untangle_roi(da, sep)

    # build the multiindex and unstack it
    da = xr.concat((da, da), 'roi')
    da = _dataarray_unstack(da, sources, targets, roi_tot, fill_value, order)

    # dataframe conversion
    if to_dataframe:
        da = _dataframe_conversion(da, order)

    return da


def conn_reshape_directed(da, sep='-', order=None, fill_value=np.nan,
                          to_dataframe=False):
    """Reshape a raveled directed array of connectivity.

    This function takes a DataArray of shape (n_pairs, n_directions) or
    (n_pairs, n_times, n_direction) where n_pairs reflects pairs of roi
    (e.g 'roi_1-roi_2') and n_direction usually contains bidirected 'x->y' and
    'y->x'. At the end, this function reshape the input array so that rows
    contains the sources and columns the targets leading to a non-symmetric
    DataArray of shape (n_roi, n_roi, n_times). A typical use case for this
    function would be after computing the covariance based granger causality.

    .. warning::

        This function reshape the data inplace. If you need to conserve the
        ravel version of your connectivity array, send a copy (`da.copy()`)

    Parameters
    ----------
    da : xarray.DataArray
        Xarray DataArray of shape (n_pairs, n_times, n_directions) where
        actually the roi dimension contains the pairs (roi_1-roi_2, roi_1-roi_3
        etc.). The dimension n_directions should contains the dimensions 'x->y'
        and 'y->x'
    sep : string | '-'
        Separator used to separate the pairs of roi names.
    order : list | None
        List of roi names to reorder the output.
    fill_value : float | np.nan
        Value to use for filling missing pairs (e.g diagonal)
    to_dataframe : bool | False
        Dataframe conversion. Only possible if the da input does not contains
        a time axis.

    Returns
    -------
    da_out : xarray.DataArray
        DataArray of shape (n_roi, n_roi, n_times)

    See also
    --------
    conn_covgc
    """
    assert isinstance(da, xr.DataArray)
    assert ('roi' in list(da.dims)) and ('direction' in list(da.dims))
    if 'times' not in list(da.dims):
        da = da.expand_dims("times")

    # get sources, targets names and sorted full list
    sources, targets, roi_tot = _untangle_roi(da, sep)

    # transpose, reindex and reorder (if needed)
    da_xy, da_yx = da.sel(direction='x->y'), da.sel(direction='y->x')
    da = xr.concat((da_xy, da_yx), 'roi')
    da = _dataarray_unstack(da, sources, targets, roi_tot, fill_value, order)

    # dataframe conversion
    if to_dataframe:
        da = _dataframe_conversion(da, order)

    return da


def _untangle_roi(da, sep):
    """Get details about the roi."""
    # start by extrating sources / targets names
    sources, targets = [], []
    for k in da['roi'].data:
        sources += [k.split(sep)[0]]
        targets += [k.split(sep)[1]]

    # merge sources and targets to force square matrix
    roi_tot = sources + targets
    _, u_idx = np.unique(roi_tot, return_index=True)
    roi_tot = np.array(roi_tot)[np.sort(u_idx)]

    return sources, targets, roi_tot


def _dataarray_unstack(da, sources, targets, roi_tot, fill_value, order):
    """Unstack a 1d to 2d DataArray."""
    import pandas as pd

    da['roi'] = pd.MultiIndex.from_arrays(
        [sources + targets, targets + sources], names=['sources', 'targets'])
    da = da.unstack(fill_value=fill_value)

    # transpose, reindex and reorder (if needed)
    da = da.transpose('sources', 'targets', 'times')
    da = da.reindex(dict(sources=roi_tot, targets=roi_tot),
                    fill_value=fill_value)
    if isinstance(order, (list, np.ndarray)):
        da = da.reindex(dict(sources=order, targets=order))

    return da


def _dataframe_conversion(da, order):
    """Convert a DataArray to a DataFrame and be sure its sorted correctly."""
    assert da.data.squeeze().ndim == 2, (
        "Dataframe conversion only possible for connectivity arrays when "
        "time dimension is missing")
    da = da.squeeze().to_dataframe('mi').reset_index()
    da = da.pivot('sources', 'targets', 'mi')
    if isinstance(order, (list, np.ndarray)):
        da = da.reindex(order, axis='index').reindex(order, axis='columns')

    return da
