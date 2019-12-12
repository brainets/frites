"""Cluster related functions."""
import logging

import numpy as np

from mne.stats.cluster_level import _find_clusters
from mne.stats import fdr_correction, bonferroni_correction

logger = logging.getLogger('frites')


def temporal_clusters_permutation_test(x, x_p, th, tail=1, mcp='maxstat',
                                       **kwargs):
    """Infer p-values using nonparametric test on temporal clusters.

    Parameters
    ----------
    x : array_like
        Array of true effect size of shape (n_roi, n_times)
    x_p : array_like
        Array of permutations of shape (n_perm, n_roi, n_times)
    th : float
        The threshold to use
    tail : {-1, 0, 1}
        Type of comparison. Use -1 for the lower part of the distribution,
        1 for the higher part and 0 for both
    mcp : {'maxstat', 'fdr', 'bonferroni'}
        Method to use for correcting p-values for the multiple comparison
        problem. By default, the maximum cluster-size across time and space is
        used.
    kwargs : dict | {}
        Additional arguments are send to the
        :func:`mne.stats.cluster_level._find_clusters`

    Returns
    -------
    pvalues : array_like
        Array of p-values of shape (n_roi, n_times)
    """
    # get variables
    n_perm, n_roi, n_times = x_p.shape
    assert tail in [-1, 0, 1]
    assert mcp in ['maxstat', 'fdr', 'bonferroni']
    kwargs['tail'] = tail

    logger.info(f"    Cluster detection (threshold={th}; mcp={mcp}; "
                f"tail={tail})")

    # -------------------------------------------------------------------------
    # identify clusters for the true x
    cl_loc, cl_mass = [], []
    for r in range(n_roi):
        _cl_loc, _cl_mass = _find_clusters(x[r, :], th, **kwargs)
        # for non-tfce, clusters are returned as a list of tuples
        _cl_loc = [k[0] if isinstance(k, tuple) else k for k in _cl_loc]
        # update cluster mass according to the tail
        if tail == 0:
            np.abs(_cl_mass, out=_cl_mass)
        elif tail == -1:
            if not isinstance(th, dict):
                _cl_mass *= -1
        # save where clusters have been found and cluster size
        cl_loc += [_cl_loc]
        cl_mass += [_cl_mass]

    # -------------------------------------------------------------------------
    # identify clusters for the permuted x
    cl_p_mass = []
    for r in range(n_roi):
        _cl_p_null = []
        for p in range(n_perm):
            _, __cl_p_null = _find_clusters(x_p[p, r, :], th, **kwargs)
            # if no cluster have been found, set a cluster mass of 0
            if not len(__cl_p_null): __cl_p_null = [0]  # noqa
            # Max / Min cluster size across time
            if tail == 1:  # max cluster size across time
                _cl_p_null += [np.r_[tuple(__cl_p_null)].max()]
            elif tail == -1:  # min cluster size across time
                if isinstance(th, dict):
                    _cl_p_null += [np.r_[tuple(__cl_p_null)].max()]
                else:
                    _cl_p_null += [np.r_[tuple(__cl_p_null)].min()]
            elif tail == 0:  # max of absolute cluster size across time
                _cl_p_null += [np.abs(np.r_[tuple(__cl_p_null)]).max()]
        cl_p_mass += [_cl_p_null]
    # array conversion of shape (n_roi, n_perm)
    cl_p_mass = np.asarray(cl_p_mass)

    # for maximum statistics, repeat the max across ROI
    if mcp == 'maxstat':
        cl_p_mass = np.tile(cl_p_mass.max(axis=0, keepdims=True), (n_roi, 1))
        pv = _clusters_to_pvalues(n_roi, n_times, n_perm, cl_loc, cl_mass,
                                  cl_p_mass)
        pv = np.clip(pv, 1. / n_perm, 1.)
    else:
        pv = _clusters_to_pvalues(n_roi, n_times, n_perm, cl_loc, cl_mass,
                                  cl_p_mass)
        if mcp is 'fdr':
            pv = fdr_correction(pv, 0.05)[1]
        if mcp is 'bonferroni':
            pv = bonferroni_correction(pv, 0.05)[1]
    pv = np.clip(pv, 0., 1.)

    return pv


def _clusters_to_pvalues(n_roi, n_times, n_perm, cl_loc, cl_mass, cl_p_mass):
    """Transform clusters into p-values."""
    pvalues = np.full((n_roi, n_times), 1.)
    for r, (cl_g, clm_g) in enumerate(zip(cl_loc, cl_mass)):
        for cl, clm in zip(cl_g, clm_g):
            pv = (clm <= cl_p_mass[[r], :]).sum(1) / n_perm
            pvalues[r, cl] = pv
    return pvalues


def cluster_threshold(x, x_p, alpha=.05, tail=1, tfce=False, n_steps=100):
    """Threshold detection for cluster-based inferencse.

    Parameters
    ----------
    x : array_like
        Array of true effect size
    x_p : array_like
        Array of permutations
    alpha : float | .05
        Thresholding permutation distribution. If `alpha` is 0.05 it means
        that the threshold is going to be the 95th percentile
    tail : {-1, 0, 1}
        Type of comparison. Use -1 for the lower part of the distribution,
        1 for the higher part and 0 for both
    tfce : bool | False
        Use Threshold Free Cluster Enhancement
    n_steps : int | 100
        Number of integration steps between the start and stoping values for
        the TFCE

    Returns
    -------
    threshold : float, dict
        The cluster threshold. For the TFCE, this output a is dictionary with
        a 'start' and a 'step' keys (MNE convention)
    """
    logger.info(f"    Cluster forming threshold (tail={tail}; alpha={alpha}; "
                f"tfce={tfce})")
    kw = dict(interpolation='nearest')
    if tfce:
        if tail == 1:
            start = max(np.percentile(x_p, 100. * (1. - alpha), **kw), 0.)
            stop = x.max()
        elif tail == -1:
            start = min(np.percentile(x_p, 100. * alpha, **kw), 0.)
            stop = x.min()
        elif tail == 0:
            start = np.percentile(np.abs(x_p), 100. * (1. - alpha), **kw)
            stop = np.abs(x).max()
        step = (stop - start) / n_steps
        th = dict(start=start, step=step)
    else:
        if tail == 1:
            th = np.nanpercentile(x_p, 100. * (1. - alpha), **kw)
        elif tail == -1:
            th = np.nanpercentile(x_p, 100. * alpha, **kw)
        elif tail == 0:
            th = np.nanpercentile(np.abs(x_p), 100. * (1. - alpha), **kw)

    return th
