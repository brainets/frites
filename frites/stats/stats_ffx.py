"""Apply maximum statistics correction."""
import logging

import numpy as np

from mne.stats import fdr_correction, bonferroni_correction

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


def ffx_cluster_fdr(mi, mi_p, alpha=0.05):
    th_pval = np.sum(mi_p > mi, axis=0) / n_perm
    is_over_th = ~fdr_correction(th_pval, alpha)[0]
    if not np.any(is_over_th):
        logger.warning(f"No gcmi exceed the threshold at p={alpha}")
        return np.full_like(mi, np.nan)


def ffx_cluster_bonferroni(mi, mi_p, alpha=0.05):
    pass


def ffx_cluster_tfce(mi, mi_p, alpha=0.05):
    pass
