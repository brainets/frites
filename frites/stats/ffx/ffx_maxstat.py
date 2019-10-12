"""Apply maximum statistics correction."""
import logging

import numpy as np

logger = logging.getLogger("frites")


def ffx_maxstat(mi, mi_p, alpha=0.05):
    """Maximum statistics correction for fixed effect inference.

    Parameters
    ----------
    mi : array_like
        Array of mutual information of shape (n_roi, n_times)
    mi_p : array_like
        Array of permuted mutual information of shape (n_perm, n_roi, n_times)
    alpha : float | 0.05
        Significiency level

    Returns
    -------
    pvalues : array_like
        Array of p-values of shape (n_roi, n_times)

    References
    ----------
    Holmes et al., 1996 :cite:`holmes1996nonparametric`
    Nichols and Holmes, 2002 :cite:`nichols2002nonparametric`
    """
    logger.info(f"    FFX maximum statistics (alpha={alpha})")
    # prepare variables
    n_perm = mi_p.shape[0]
    pvalues = np.full_like(mi, 1.)
    # value to use as the threshold
    p_max = np.percentile(mi_p.max(axis=(1, 2)), 100. * (1. - alpha),
                          interpolation='higher')
    # infer p-values
    pvalues[mi > p_max] = alpha

    return pvalues


def ffx_cluster_fdr():
    pass


def ffx_cluster_bonferroni():
    pass


def ffx_cluster_tfce():
    pass
