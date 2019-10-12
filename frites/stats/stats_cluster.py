"""Cluster related functions."""
import numpy as np

from mne.stats.cluster_level import _find_clusters


def find_temporal_clusters(mi, mi_p, th, tail=1, **kwargs):
    """Find clusters in the array of true mi.

    Parameters
    ----------
    mi : array_like
        Array of true mutual information of shape (n_roi, n_times)
    mi_p : array_like
        Array of permuted mutual information of shape (n_perm, n_roi, n_times)
    th : float
        The threshold to use
    tail : {-1, 0, 1}
        Type of comparison. Use -1 for the lower part of the distribution,
        1 for the higher part and 0 for both
    kwargs : dict | {}
        Additional arguments are send to the
        :func:`mne.stats.cluster_level._find_clusters`

    Returns
    -------
    pvalues : array_like
        Array of p-values of shape (n_roi, n_times)
    """
    # get variables
    n_perm, n_roi, n_times = mi_p.shape
    kwargs['tail'] = kwargs.get('tail', 1)

    # -------------------------------------------------------------------------
    # identify clusters for the true mi
    cl_true, cl_mass = [], []
    for r in range(n_roi):
        _cl_true, _cl_mass = _find_clusters(mi[r, :], th, **kwargs)
        cl_true += [_cl_true]
        cl_mass += [_cl_mass]

    # -------------------------------------------------------------------------
    # identify clusters for the permuted mi
    cl_p_mass = []
    for p in range(n_perm):
        _cl_p_null = []
        for r in range(n_roi):
            _, __cl_p_null = _find_clusters(mi_p[p, r, :], th, **kwargs)
            # if no cluster have been found, set a cluster mass of 0
            if not len(__cl_p_null): __cl_p_null = [0]  # noqa
            _cl_p_null += [__cl_p_null]
        # if this roi has no clusters, just ignore it
        if not len(_cl_p_null): continue  # noqa
        # take the maximum size (MCP)
        cl_p_mass += [np.r_[tuple(_cl_p_null)].max()]
    cl_p_mass = np.array(cl_p_mass)

    # -------------------------------------------------------------------------
    # infer p-values by comparing cluster sizes
    pvalues = np.full((n_roi, n_times), 1.)
    for r, (cl_g, clm_g) in enumerate(zip(cl_true, cl_mass)):
        for cl, clm in zip(cl_g, clm_g):
            pv = (clm <= cl_p_mass.reshape(1, -1)).sum(1) / n_perm
            pvalues[r, cl[0]] = max(1. / n_perm, pv)

    return pvalues
