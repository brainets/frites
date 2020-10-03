"""Utility functions for connectivity."""
import numpy as np
import xarray as xr


def conn_reshape_undirected(da, sep='-', order=None):
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

    Returns
    -------
    da_out : xarray.DataArray
        DataArray of shape (n_roi, n_roi, n_times)
    """
    import pandas as pd

    assert isinstance(da, xr.DataArray)
    assert 'roi' in list(da.dims)

    if 'times' not in list(da.dims):
        da = da.expand_dims("times")

    # start by extrating sources / targets names
    sources, targets = [], []
    for k in da['roi'].data:
        sources += [k.split(sep)[0]]
        targets += [k.split(sep)[1]]

    # build the multiindex and unstack it
    da['roi'] = pd.MultiIndex.from_arrays(
        [sources, targets], names=['sources', 'targets'] )
    da = da.unstack(fill_value=0).transpose('sources', 'targets', 'times')

    # force square matrix
    roi_tot = sources + targets
    _, u_idx = np.unique(roi_tot, return_index=True)
    roi_tot = np.array(roi_tot)[np.sort(u_idx)]
    da = da.reindex(dict(sources=roi_tot, targets=roi_tot), fill_value=0.)

    # reorder (if needed)
    if isinstance(order, (list, np.ndarray)):
        da = da.reindex(dict(sources=order, targets=order))

    # make it symetric and fill diagonal with nan
    da.data += da.transpose('targets', 'sources', 'times').data
    idx_diag = np.diag_indices(da.shape[0])
    da.data[idx_diag[0], idx_diag[1], :] = np.nan

    return da
