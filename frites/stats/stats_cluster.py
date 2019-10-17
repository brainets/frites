"""Cluster related functions."""
import numpy as np

from mne.stats.cluster_level import _find_clusters


def temporal_clusters_permutation_test(mi, mi_p, th, tail=1, **kwargs):
    """Infer p-values using nonparametric test on temporal clusters.

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
    assert tail in [-1, 0, 1]
    kwargs['tail'] = tail

    # -------------------------------------------------------------------------
    # identify clusters for the true mi
    cl_true, cl_mass = [], []
    for r in range(n_roi):
        _cl_true, _cl_mass = _find_clusters(mi[r, :], th, **kwargs)
        # for non-tfce, clusters are returned as a list of tuples
        _cl_true = [k[0] if isinstance(k, tuple) else k for k in _cl_true]
        # save where clusters have been found and cluster size
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
    cl_p_mass = np.array(cl_p_mass).reshape(1, -1)

    # -------------------------------------------------------------------------
    # infer p-values by comparing cluster sizes
    pvalues = np.full((n_roi, n_times), 1.)
    for r, (cl_g, clm_g) in enumerate(zip(cl_true, cl_mass)):
        for cl, clm in zip(cl_g, clm_g):
            if tail == 1:
                pv = (clm <= cl_p_mass).sum(1) / n_perm
            elif tail == -1:
                pv = (clm >= cl_p_mass).sum(1) / n_perm
            elif tail == 0:
                pv = (np.abs(clm) <= np.abs(cl_p_mass)).sum(1) / n_perm
            pvalues[r, cl] = max(1. / n_perm, pv)

    return pvalues
