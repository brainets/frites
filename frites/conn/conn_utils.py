"""Utility functions for connectivity."""
import numpy as np
import xarray as xr
import pandas as pd

from frites.utils import nonsorted_unique
from frites.io import set_log_level, logger


###############################################################################
###############################################################################
#                                 CONN PAIRS
###############################################################################
###############################################################################


def conn_get_pairs(roi, directed=False, nb_min_suj=-np.inf, verbose=None):
    """Get possible connectivity pairs for multiple subjects.

    This function returns a DataFrame that contains all of the necessary
    informations for managing pairs of brain regions across many subjects.

    Parameters
    ----------
    roi : list
        List where each item in this list is an array descriving the brain
        region names of a single subject.
    directed : bool | False
        Specify whether the the returned pairs should be for directed (True)
        or undirected (default : False) connectivity.
    nb_min_suj : int | -np.inf
        Specify whether the pairs should be represented by a minimum number of
        subjects.

    Returns
    -------
    df_conn : pd.DataFrame
        A Pandas DataFrame that describes the connectivity informations at the
        group level. The table contains the following entries :

            * 'sources' / 'targets' : respectively, the source and target names
            * 'subjects' : list of subjects per pair of brain regions
            * '#subjects' : number of subjects per pair of brain regions
            * 'names' : name of each pair. If undirected, the names are going
              to be like 'roi_0-roi_1' or 'roi_0->roi_1' if directed
            * 'keep' : booleans indicating whether the number of subjects per
              pair of brain regions is over nb_min_suj
    df_suj : pd.DataFrame
        A Pandas DataFrame that describes the connectivity information per
        subject. The table contains the following entries :

            * 'subjects' : subject number
            * 'keep_roi' / 'drop_roi' : the brain regions respectively to keep
              and to remove to fit the input parameters nb_min_suj
            * 'keep_suj' : boolean describing if the subject should be dropped
              or conserved
            * 'conn' : the 2D boolean connectivity array per subject
    """
    set_log_level(verbose)
    assert isinstance(roi, list)
    n_subjects = len(roi)
    roi = [np.asarray(k) for k in roi]

    # =========================== Conn info per pair ==========================

    s_ss, t_ss, ss = [], [], []
    for k in range(n_subjects):
        # get the unique list of unsorted list of brain regions
        u_roi = nonsorted_unique(roi[k], assert_unique=True)
        n_u_roi = len(u_roi)
        # get all possible pairs
        if directed:
            pairs = np.where(~np.eye(n_u_roi, dtype=bool))
        else:
            pairs = np.triu_indices(n_u_roi, k=1)
        s_names, t_names = u_roi[pairs[0]], u_roi[pairs[1]]
        # if not directed, merge '0-1' and '1-0'
        if not directed:
            st_names = np.c_[s_names, t_names]
            s_names, t_names = np.unique(np.sort(st_names, axis=1), axis=0).T
        # keep single-subject source and target names
        s_ss += [s_names]
        t_ss += [t_names]
        ss += [k] * len(s_names)
    # fill info in a dataframe
    df_ss = pd.DataFrame({
        'subjects': ss,
        'sources': np.concatenate(s_ss),
        'targets': np.concatenate(t_ss)
    })

    # get the number of subjects per pair
    pattern = '->' if directed else '-'
    gp = df_ss.groupby(['sources', 'targets'])
    df_conn = gp.subjects.aggregate([list]).reset_index()
    df_conn = df_conn.rename(columns={'list': 'subjects'})
    df_conn['#subjects'] = [len(k) for k in df_conn['subjects']]
    df_conn['names'] = [f"{k}{pattern}{i}" for k, i in zip(
        df_conn['sources'], df_conn['targets'])]
    df_conn['keep'] = df_conn['#subjects'] >= nb_min_suj

    # print the info
    n_remain = np.sum(list(df_conn['keep']))
    n_drop = np.sum(list(~df_conn['keep']))
    logger.info(f"    {n_remain} remaining pairs of brain regions "
                f"(nb_min_suj={nb_min_suj}), {n_drop} dropped")

    # ========================= Conn info per subject =========================

    # build 2d connectivity array per subject
    conn = {}
    for n_s in range(n_subjects):
        n_roi_s = len(roi[n_s])
        _conn = xr.DataArray(
            ~np.eye(n_roi_s, dtype=bool), dims=('sources', 'targets'),
            coords=(roi[n_s], roi[n_s]))
        conn[n_s] = _conn

    # fill the information
    for k in range(len(df_conn)):
        _df = df_conn.iloc[k, :]
        for s in _df['subjects']:
            _s, _t, _k = _df['sources'], _df['targets'], bool(_df['keep'])
            conn[s].loc[dict(sources=_s, targets=_t)] = _k
            if not directed:
                conn[s].loc[dict(sources=_t, targets=_s)] = _k

    # get the brain regions to keep / drop per subject
    suj, roi_keep, roi_drop, conn_tot = [], [], [], []
    for s in range(n_subjects):
        _keep = roi[s][np.union1d(*np.where(conn[s]))]
        _drop = np.setdiff1d(roi[s], _keep)
        suj += [s]
        roi_keep += [_keep.tolist()]
        roi_drop += [_drop.tolist()]
        conn_tot += [conn[s].data]
    # create the final dataframe
    df_suj = pd.DataFrame({'subjects': suj, 'keep_roi': roi_keep,
                           'drop_roi': roi_drop})  # , 'conn': conn_tot
    df_suj['keep_suj'] = [len(k) > 1 for k in df_suj['keep_roi']]

    return df_conn, df_suj


def conn_links(roi, directed=False, net=False, roi_relation='both', sep='auto',
               nb_min_links=None, pairs=None, sort=True, triu_k=1,
               hemisphere=None, hemi_links='both', categories=None,
               source_seed=None, target_seed=None, verbose=None):
    """Construct pairwise links for functional connectivity.

    This function can be used for defining the pairwise links for computing
    either undirected or directed FC on M/EEG or intracranial EEG.

    Parameters
    ----------
    roi : array_like
        List of roi (or contacts) names.
    directed : bool | False
        Specify whether the links should be for undirected (False) or directed
        (True) FC
    net : bool | False
        Specify whether it is for net directed FC (True) or not (False)
    roi_relation : {'both', 'inter', 'intra'}
        Specify the roi relation between pairs of brain regions. Use either :

            * 'intra' : to select only links within a brain region
              (e.g. V1-V1)
            * 'inter' : to select only links across brain region (e.g. V1-V2)
            * 'both' : to select links both within and across brain regions
    sep : string | 'auto'
        If 'auto', '-' are used to linked brain region names for undirected FC
        or '->' for directed FC. Alternatively, you can provide a custom
        separator (e.g. sep='/')
    nb_min_links : int | None
        Threshold for defining a minimum number of links between two brain
        regions (e.g. iEEG)
    pairs : array_like | None
        Force to use certain pairs of brain regions. Should be an array of
        shape (n_pairs, 2) where the first column refer to sources and the
        second to targets
    sort : bool | True
        For undirected and net directed FC, sort the names of the brain regions
        (e.g. 'V1-M1' -> 'M1-V1')
    triu_k : int | 1
        Diagonal offset when estimating the undirected links to use. By
        default, triu_k=1 means that we skip auto-connections
    hemisphere : array_like | None
        List of hemisphere names
    hemi_links : {'both', 'intra', 'inter'}
        Specify whether connectivity links should be :

            * 'both': intra-hemispheric and inter-hemispheric (default)
            * 'intra': intra-hemispheric
            * 'inter': inter-hemispheric

        In order to work, you should provide the hemisphere name using the
        input `hemisphere`
    categories : array_like | list | None
        Specify categorical information associated to each region to then
        get links only across categories.
    source_seed : str, list | None
        Brain region name(s) to use as source seed. Can either be the name of a
        single brain region or a list of brain regions.
    target_seed : str, list | None
        Brain region name(s) to use as target seed. Can either be the name of a
        single brain region or a list of brain regions.

    Returns
    -------
    indices : tuple
        Remaining indices for (sources, targets)
    roi_st : list
        Name of the pairs of brain regions
    """
    set_log_level(verbose)
    assert isinstance(roi, (np.ndarray, list, tuple))
    if not directed:
        assert not net, ("Net computations not supported for undirected "
                         "connectivity")
    roi = np.asarray(roi).astype(str)
    n_roi = len(roi)
    if isinstance(source_seed, str): source_seed = [source_seed]  # noqa
    if isinstance(target_seed, str): target_seed = [target_seed]  # noqa
    logger.info(f"Defining links (n_roi={n_roi}; directed={directed}; "
                f"net={net}, nb_min_links={nb_min_links})")

    # build separator name
    if sep == 'auto':
        sep = '->' if directed and not net else '-'
    else:
        assert isinstance(sep, str)

    # get (un)directed pairs
    if isinstance(pairs, np.ndarray) and (pairs.shape[1] == 2):
        x_s, x_t = pairs[:, 0], pairs[:, 1]
    else:
        if directed and not net:
            x_s, x_t = np.where(~np.eye(n_roi, dtype=bool))
        elif (not directed) or (directed and net):
            x_s, x_t = np.triu_indices(n_roi, k=triu_k)

    # manage roi relation
    if roi_relation in ['inter', 'intra']:
        logger.info(f"    Keeping only {roi_relation}-roi links")
        roi_s, roi_t = roi[x_s], roi[x_t]
        if roi_relation == 'intra':
            keep = [s == t for s, t in zip(roi_s, roi_t)]
        elif roi_relation == 'inter':
            keep = [s != t for s, t in zip(roi_s, roi_t)]
        keep = np.asarray(keep)

        logger.info(f"    ROI relation selection (dropped={(~keep).sum()} "
                    "links)")
        x_s, x_t = x_s[keep], x_t[keep]

    # change roi order for undirected and net directed
    if sort and (not directed) or (directed and net):
        logger.info("    Sorting roi names")
        roi_low = np.asarray([np.char.lower(r.astype(str)) for r in roi])
        _xs, _xt = x_s.copy(), x_t.copy()
        x_s, x_t = [], []
        for s, t in zip(_xs, _xt):
            _pair = np.array([roi_low[s], roi_low[t]])
            if np.all(_pair == np.sort(_pair)):
                x_s.append(s)
                x_t.append(t)
            else:
                x_s.append(t)
                x_t.append(s)
    x_s, x_t = np.asarray(x_s), np.asarray(x_t)

    # keep pairs with a minimum number of links inside
    if isinstance(nb_min_links, int):
        logger.info("    Thresholding number of links")
        roi_st = [f"{s}{sep}{t}" for s, t in zip(roi[x_s], roi[x_t])]
        df = pd.DataFrame({'pairs': roi_st})
        df = df.groupby('pairs').size()
        keep = [df.loc[r] >= nb_min_links for r in roi_st]
        x_s, x_t = x_s[keep], x_t[keep]

    # hemisphere selection
    if isinstance(hemisphere, (list, np.ndarray)):
        assert hemi_links in ['both', 'intra', 'inter']
        hemisphere = np.asarray(hemisphere)
        h_s, h_t = hemisphere[x_s], hemisphere[x_t]
        if hemi_links in ['intra', 'inter']:
            keep = h_s == h_t if hemi_links == 'intra' else h_s != h_t
            x_s, x_t = x_s[keep], x_t[keep]
        else:
            keep = np.array([True] * len(x_s))
        logger.info(f"    Hemispheric selection (hemi_links={hemi_links}, "
                    f"dropped={(~keep).sum()} links)")

    # categorical selection
    if isinstance(categories, (list, np.ndarray, tuple)):
        # reshape categories
        categories = np.asarray(categories)
        if categories.ndim == 1:
            categories = categories[:, np.newaxis]
        assert categories.shape[0] == n_roi

        # categorical selection
        keep = []
        for s, t in zip(x_s, x_t):
            keep.append(np.all(categories[s, :] != categories[t, :]))
        keep = np.asarray(keep)
        x_s, x_t = x_s[keep], x_t[keep]

        logger.info(f"    Categorical selection (dropped={(~keep).sum()} "
                    "links)")

    # seed / target selection
    if isinstance(source_seed, (list, tuple, np.ndarray)):
        keep = _seed_selection(source_seed, roi, x_s, x_t, directed, 'source')
        x_s, x_t = x_s[keep], x_t[keep]
    if isinstance(target_seed, (list, tuple, np.ndarray)):
        keep = _seed_selection(target_seed, roi, x_s, x_t, directed, 'target')
        x_s, x_t = x_s[keep], x_t[keep]

    # build pairs of brain region names
    roi_st = np.asarray([f"{s}{sep}{t}" for s, t in zip(roi[x_s], roi[x_t])])

    return (x_s, x_t), roi_st


def _seed_selection(seed, roi, x_s, x_t, directed, origin):
    if directed:
        if origin == 'source':
            keep = [s in seed for s in roi[x_s]]
        elif origin == 'target':
            keep = [s in seed for s in roi[x_t]]
    else:
        keep_s = [s in seed for s in roi[x_s]]
        keep_t = [t in seed for t in roi[x_t]]
        keep = np.c_[keep_s, keep_t].any(1)
    return keep

###############################################################################
###############################################################################
#                              CONN RESHAPING
###############################################################################
###############################################################################


def conn_reshape_undirected(da, sep='-', order=None, rm_missing=False,
                            fill_value=np.nan, fill_diagonal=None,
                            to_dataframe=False, inplace=False, verbose=None):
    """Reshape a raveled undirected array of connectivity.

    This function takes a DataArray of shape (n_pairs,) or (n_pairs, n_times)
    where n_pairs reflects pairs of roi (e.g 'roi_1-roi_2') and reshape it to
    be a symmetric DataArray of shape (n_roi, n_roi, n_times).

    Parameters
    ----------
    da : xarray.DataArray
        Xarray DataArray of shape (n_pairs, n_times) where actually the roi
        dimension contains the pairs (roi_1-roi_2, roi_1-roi_3 etc.)
    sep : string | '-'
        Separator used to separate the pairs of roi names.
    order : list | None
        List of roi names to reorder the output.
    rm_missing : bool | False
        When reordering the connectivity array, choose if you prefer to reindex
        even if there's missing regions (rm_missing=False) or if missing
        regions should be removed (rm_missing=True)
    fill_value : float | np.nan
        Value to use for filling missing pairs
    fill_diagonal : float | None
        Value to use in order to fill the diagonal. If None, the diagonal is
        untouched
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
    set_log_level(verbose)
    assert isinstance(da, xr.DataArray)
    if not inplace:
        da = da.copy()
    assert 'roi' in list(da.dims)
    if 'times' not in list(da.dims):
        da = da.expand_dims("times")

    # get sources, targets names and sorted full list
    sources, targets, roi_tot = _untangle_roi(da, sep)

    # duplicates to make it symmetrical
    da = xr.concat((da, da), 'roi')
    s_, t_ = sources + targets, targets + sources
    # build the multiindex and unstack it
    da, order = _dataarray_unstack(da, s_, t_, roi_tot, fill_value,
                                   order, rm_missing, fill_diagonal)

    # dataframe conversion
    if to_dataframe:
        da = _dataframe_conversion(da, order, rm_missing)

    return da


def conn_reshape_directed(da, net=False, sep='-', order=None, rm_missing=False,
                          fill_value=np.nan, fill_diagonal=None,
                          to_dataframe=False, inplace=False, verbose=None):
    """Reshape a raveled directed array of connectivity.

    This function takes a DataArray of shape (n_pairs, n_directions) or
    (n_pairs, n_times, n_direction) where n_pairs reflects pairs of roi
    (e.g 'roi_1-roi_2') and n_direction usually contains bidirected 'x->y' and
    'y->x'. At the end, this function reshape the input array so that rows
    contains the sources and columns the targets leading to a non-symmetric
    DataArray of shape (n_roi, n_roi, n_times). A typical use case for this
    function would be after computing the covariance based granger causality.

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
    rm_missing : bool | False
        When reordering the connectivity array, choose if you prefer to reindex
        even if there's missing regions (rm_missing=False) or if missing
        regions should be removed (rm_missing=True)
    fill_value : float | np.nan
        Value to use for filling missing pairs (e.g diagonal)
    fill_diagonal : float | None
        Value to use in order to fill the diagonal. If None, the diagonal is
        untouched
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
    set_log_level(verbose)
    assert isinstance(da, xr.DataArray)
    if not inplace:
        da = da.copy()
    assert 'roi' in list(da.dims)
    if 'times' not in list(da.dims):
        da = da.expand_dims("times")

    # get sources, targets names and sorted full list
    sources, targets, roi_tot = _untangle_roi(da, sep)

    # transpose, reindex and reorder (if needed)
    if 'direction' in list(da.dims):
        da_xy, da_yx = da.sel(direction='x->y'), da.sel(direction='y->x')
        if net:
            da = xr.concat((da_xy - da_yx, da_xy - da_yx), 'roi')
        else:
            da = xr.concat((da_xy, da_yx), 'roi')
        s_, t_ = sources + targets, targets + sources
    else:
        s_, t_ = sources, targets
    da, order = _dataarray_unstack(da, s_, t_, roi_tot, fill_value,
                                   order, rm_missing, fill_diagonal)

    # dataframe conversion
    if to_dataframe:
        da = _dataframe_conversion(da, order, rm_missing)

    return da


def _untangle_roi(da, sep):
    """Get details about the roi."""
    # start by extrating sources / targets names
    sources, targets = [], []
    for k in da['roi'].data:
        sources += [k.split(sep)[0]]
        targets += [k.split(sep)[1]]

    # merge sources and targets to force square matrix
    roi_tot = nonsorted_unique(sources + targets)

    return sources, targets, roi_tot


def _dataarray_unstack(da, sources, targets, roi_tot, fill_value, order,
                       rm_missing, fill_diagonal):
    """Unstack a 1d to 2d DataArray."""
    import pandas as pd

    # build the multi-index
    da['roi'] = pd.MultiIndex.from_arrays(
        [sources, targets], names=['sources', 'targets'])
    # test for duplicated entries
    st_names = pd.Series([f"{s}-{t}" for s, t in zip(sources, targets)])
    duplicates = np.array(list(st_names.duplicated(keep='first')))
    if duplicates.any():
        logger.warning(f"Duplicated entries found and removed : "
                       f"{da['roi'].data[duplicates]}")
        da = da.sel(roi=~duplicates)
    # unstack to be 2D/3D
    da = da.unstack(fill_value=fill_value)

    # transpose, reindex and reorder (if needed)
    da = da.transpose('sources', 'targets', 'times')
    da = da.reindex(dict(sources=roi_tot, targets=roi_tot),
                    fill_value=fill_value)

    # change order
    if isinstance(order, (list, np.ndarray)):
        if rm_missing:
            order = [k for k in order.copy() if k in roi_tot.tolist()]
        da = da.reindex(dict(sources=order, targets=order))

    # fill diagonal (if needed)
    if fill_diagonal is not None:
        di = np.diag_indices(da.shape[0])
        da.data[di[0], di[1], :] = fill_diagonal

    return da, order


def _dataframe_conversion(da, order, rm_missing):
    """Convert a DataArray to a DataFrame and be sure its sorted correctly."""
    assert da.data.squeeze().ndim == 2, (
        "Dataframe conversion only possible for connectivity arrays when "
        "time dimension is missing")
    da = da.squeeze().to_dataframe('mi').reset_index()
    da = da.pivot('sources', 'targets', 'mi')
    if isinstance(order, (list, np.ndarray)):
        da = da.reindex(order, axis='index').reindex(order, axis='columns')
    # drop empty lines
    if rm_missing:
        da = da.dropna(axis=0, how='all').dropna(axis=1, how='all')

    return da


def conn_ravel_directed(da, sep='-', drop_within=False):
    """Ravel a directed array.

    This function reorganize a directed array that contains the coordinates
    x->y and y->x to a single coordinate 'x->y'.

    Parameters
    ----------
    da : xarray.DataArray
        Xarray DataArray that should at least contains the dimensions 'roi'
        and 'direction'. The dimension 'direction' should also contains the
        coordinates 'x->y' and 'y->x'
    sep : string | '-'
        Separator used to separate the pairs of roi names.
    drop_within : bool | False
        Drop within node connections

    Returns
    -------
    da_r : xarray.DataArray
        Raveled array of directed connectivity
    """
    # inputs testing
    assert isinstance(da, xr.DataArray) and isinstance(sep, str)
    assert 'direction' in da.dims, "Should be a directed array"
    assert 'roi' in da.dims, "Missing roi dimension"
    directions = da['direction'].data
    assert ('x->y' in directions) and ('y->x' in directions)

    # build bidirected roi
    roi_xy, roi_yx = [], []
    for r in da['roi'].data:
        r_s, r_t = r.split(sep)
        roi_xy.append(f"{r_s}->{r_t}")
        roi_yx.append(f"{r_t}->{r_s}")

    # select bidirected arrays
    da_xy = da.sel(direction='x->y').drop_vars('direction')
    da_yx = da.sel(direction='y->x').drop_vars('direction')

    # replace roi names
    da_xy['roi'] = roi_xy
    da_yx['roi'] = roi_yx

    # finally, concat both
    da_ravel = xr.concat((da_xy, da_yx), 'roi')

    # drop within node connections
    if drop_within:
        to_keep = []
        for r in da_ravel['roi'].data:
            r_s, r_t = r.split('->')
            to_keep.append(r_s != r_t)
        da_ravel = da_ravel.sel(roi=to_keep)

    return da_ravel


def conn_net(da, roi='roi', order=None, sep='-', invert=False, verbose=None):
    """Compute the net on directed connectivity.

    This function can be used to compute the net difference on directed
    connectivity (i.e. A - B = A->B - B->A).

    Parameters
    ----------
    da : xr.DataArray
        Xarray DataArray containing the connectivity array
    roi : 'roi'
        Name of the spatial dimension
    order : list | None
        List of names for specifying the final order
    sep : string | '-'
        Separator between brain region names (e.g. if 'Insula->Thalamus' then
        sep is '->')
    invert : bool | False
        Specify whether the difference should be computed with A - B or B - A

    Returns
    -------
    out : xr.DataArray
        DataArray, with the same dimension names as the input, representing the
        net difference of directed connexions.
    """
    set_log_level(verbose)
    assert roi in da.dims
    roi_names = da[roi].data

    # get roi order from sources
    if order is None:
        roi_s, roi_t = [], []
        for r in roi_names:
            _rs, _rt = r.split(sep)
            roi_s.append(_rs)
            roi_t.append(_rt)
        order = nonsorted_unique(roi_s + roi_t)
    order = np.asarray(order)

    # build names of the difference
    x_s, x_t = np.triu_indices(len(order), k=1)
    roi_s, roi_t = order[x_s], order[x_t]
    if invert:
        _roi_st = roi_s.copy()
        roi_s = roi_t
        roi_t = _roi_st

    # build pairs names
    roi_st, p_s, p_t, ignored = [], [], [], []
    for s, t in zip(roi_s, roi_t):
        name_s, name_t = f"{s}{sep}{t}", f"{t}{sep}{s}"
        if (name_s in da[roi]) and (name_t in da[roi]):
            roi_st.append(f"{s}-{t}")
            p_s.append(name_s)
            p_t.append(name_t)
        else:
            ignored.append(f"{s}-{t}")
            # ignored.append(name_s)
    if len(ignored):
        logger.warning("The following pairs have been ignored in the "
                       f"subtraction : {ignored}")

    # prepare the output
    out = da.isel(**{roi: slice(0, len(roi_st))}).copy()
    out[roi] = roi_st
    out.data = da.sel(**{roi: p_s}).data - da.sel(**{roi: p_t}).data

    # update attributes to track operations
    out.attrs['net_source'] = p_s
    out.attrs['net_target'] = p_t
    out.name = da.name + '_net' if da.name else 'Net conn'

    return out
