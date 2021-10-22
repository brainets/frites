import numpy as np
import xarray as xr

from mne.utils import ProgressBar

from frites.io import check_attrs
from frites.estimator import CorrEstimator


def conn_fcd_corr(conn, roi='roi', times='times', tskip=1, estimator=None,
                  fill_diagonal=np.nan, verbose=None):
    """Compute the correlation on dynamic network.

    This function can be used to compute the correlation between time points
    of a dynamic functional connectivity array, namely :

    .. math::

        corr(conn_{t_{i}}, conn_{t_{j}})

    Parameters
    ----------
    conn : xr.DataArray
        3D array of dynamic functional connectivity of shape
        (n_samples, n_pairs, n_times)
    roi : string | 'roi'
        Name of the spatial dimension describing the names of the pairs of
        brain regions
    times : string | 'times'
        Name of the temporal dimension
    tskip : int | 1
        Number of time point to skip (equivalent to conn[..., ::tskip])
    estimator : frites.estimator | None
        Estimator in order to measure the amount of information shared between
        two time-series coming from two distinct brain regions. Note that if
        you want to privide an estimator, be sure that it is made for
        continuous variables (mi_type='cc'). By default the correlation is used
    fill_diagonal : float | None
        Value to use in order to fill the diagonal. By default, the diagonal is
        filled with nans

    Returns
    -------
    corr : xr.DataArray
        The correlation array of shape (n_samples, n_times, n_times)
    """
    # __________________________________ I/O __________________________________
    # check input type
    assert isinstance(conn, xr.DataArray) and (conn.ndim == 3)
    assert (roi in conn.coords) and (times in conn.coords)
    supp_dim = np.setdiff1d(conn.dims, [roi, times])[0]
    attrs = conn.attrs

    # reshape (if needed)
    if conn.dims != (supp_dim, times, roi):
        conn = conn.transpose(supp_dim, times, roi)
    conn = conn.loc[:, ::tskip, :]

    # get coordinates
    supp_c, times_c = conn[supp_dim].data, conn[times].data

    # _______________________________ ESTIMATOR _______________________________
    if estimator is None:
        estimator = CorrEstimator(verbose=verbose)
    assert estimator.settings['mi_type'] == 'cc', (
        "Estimator should extract information between two continuous "
        "variables (mi_type='cc')")
    fcn = estimator.get_function()

    # __________________________________ CORR _________________________________
    # compute the correlation
    corr = np.full((len(supp_c), len(times_c), len(times_c)), fill_diagonal)
    t_source, t_target = np.triu_indices(len(times_c), k=1)
    pbar = ProgressBar(range(len(times_c)), mesg='Estimating FCD correlation')
    for t_s in range(len(times_c)):
        data_s = conn.data[:, [t_s], :]
        for t_t in range(t_s + 1, len(times_c)):
            data_t = conn.data[:, [t_t], :]
            corr[:, t_s, t_t] = fcn(data_s, data_t)
            corr[:, t_t, t_s] = corr[:, t_s, t_t]
        pbar.update_with_increment_value(1)

    # _________________________________ XARRAY ________________________________
    # build attributes
    attrs.update({
        'estimator': estimator.name,
        'type': 'fcd_corr',
        'input': conn.name
    })
    # xarray transoformation
    corr = xr.DataArray(
        corr, dims=(supp_dim, 'times_source', 'times_target'),
        coords=(supp_c, times_c, times_c), attrs=check_attrs(attrs),
        name=f'FCD Corr ({estimator.name})'
    )

    return corr
