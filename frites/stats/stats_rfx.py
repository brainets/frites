"""Random effect functions."""
import logging

import numpy as np
from scipy.stats import trim_mean, ttest_1samp

from .stats_cluster import temporal_clusters_permutation_test

logger = logging.getLogger("frites")


RECENTER = dict(mean=np.mean, median=np.median,
                trimmed=lambda x, axis=0: trim_mean(x, .2, axis=axis))


def _rfx_ttest(mi, mi_p, center=False, zscore=False):
    """Perform the t-test across subjects."""
    n_roi = len(mi_p)
    # remove the mean / median / trimmed
    if center in RECENTER.keys():
        logger.info(f"    RFX recenter distributions (center={center}, "
                    f"z-score={zscore})")
        for k in range(len(mi)):
            _med = RECENTER[center](mi_p[k], axis=0)
            _std = mi_p[k].std(axis=0) if zscore else 1.
            mi[k] = (mi[k] - _med) / _std
            mi_p[k] = (mi_p[k] - _med) / _std
    # get the mean of surrogates
    _merge_perm = np.r_[tuple([mi_p[k].ravel() for k in range(n_roi)])]
    pop_mean_surr = np.mean(_merge_perm)
    # perform the one sample t-test against the mean both on the true and
    # permuted mi
    t_obs = np.stack([ttest_1samp(mi[k], pop_mean_surr, axis=0)[
        0] for k in range(n_roi)])
    t_obs_surr = np.stack([ttest_1samp(mi_p[k], pop_mean_surr, axis=1)[
        0] for k in range(n_roi)])
    t_obs_surr = np.swapaxes(t_obs_surr, 0, 1)

    return t_obs, t_obs_surr


def rfx_cluster_ttest(mi, mi_p, alpha=0.05, center=False, zscore=False):
    """T-test across subjects for random effect inference.

    This function performed the following steps :

        * take the mean over permutations
        * perform a one-sample t-test across the subject dimension against the
          mean of permutations. This t-test is performed both on the true and
          permuted mi
        * as a threshold, use the 100 * (1-alpha)th percentile of the t-test
          performed over the permutations
        * use this threshold to detect clusters both on the true and permuted
          mi
        * take the maximum of the permuted clusters (MCP)
        * compare cluster sizes and infer p-values

    By using a t-test over the subject dimension, we actually make the
    hypothesis that subjects are normally distributed at each time point and
    region of interest.

    Parameters
    ----------
    mi : array_like
        A list of length n_roi of array of mutual information of shape
        A list of array of permuted mutual information of shape
        (n_perm, n_suj, n_times)
    alpha : float | 0.05
        Significiency level
    center : {'mean', "median", "trimmed"} | False
        If True, substract the mean of the surrogates to the true and permuted
        mi. The median or the 20% trimmed mean can also be removed
        :cite:`wilcox2018guide`
    zscore : bool | False
        Apply z-score normalization to the true and permuted mutual information

    Returns
    -------
    pvalues : array_like
        Array of p-values of shape (n_suj, n_times)

    References
    ----------
    Giordano et al., 2017 :cite:`giordano2017contributions`
    """
    # get t-test over true and permuted mi
    t_obs, t_obs_surr = _rfx_ttest(mi, mi_p, center=center, zscore=zscore)
    # at this point, t_obs.shape is (n_roi, n_times) and t_obs_surr.shape is
    # (n_perm, n_roi, n_times). Now, infer the threshold to use for detecting
    # clusters
    perc = 100. * (1. - alpha)
    th = np.nanpercentile(t_obs_surr, perc, interpolation='nearest')
    # infer p-values
    logger.info(f"    RFX non-parametric group t-test (alpha={alpha}, "
                f"threshold={th})")
    pvalues = temporal_clusters_permutation_test(t_obs, t_obs_surr, th, tail=1)

    return pvalues


def rfx_cluster_ttest_tfce(mi, mi_p, alpha=0.05, start=None, step=None,
                           center=False, zscore=False):
    """TFCE and T-test across subjects for random effect inference.

    This function performed the following steps :

        * take the mean over permutations
        * perform a one-sample t-test across the subject dimension against the
          mean of permutations. This t-test is performed both on the true and
          permuted mi
        * threshold is defined using integration parameters (`start`, `step`)
        * use this threshold to detect clusters both on the true and permuted
          mi
        * take the maximum of the permuted clusters (MCP)
        * compare cluster sizes and infer p-values

    By using a t-test over the subject dimension, we actually make the
    hypothesis that subjects are normally distributed at each time point and
    region of interest.

    Parameters
    ----------
    mi : array_like
        Array of mutual information of shape (n_roi, n_times)
    mi_p : array_like
        Array of permuted mutual information of shape (n_perm, n_roi, n_times)
    alpha : float | 0.05
        Significiency level
    start : int, float | None
        Starting point for the TFCE integration. If None, `start` is going to
        be set to 0
    step : int, float | None
        Step for the TFCE integration. If None, `step` is going to be defined
        in order to have 100 steps
    center : {'mean', "median", "trimmed"} | False
        If True, substract the mean of the surrogates to the true and permuted
        mi. The median or the 20% trimmed mean can also be removed
        :cite:`wilcox2018guide`
    zscore : bool | False
        Apply z-score normalization to the true and permuted mutual information

    Returns
    -------
    pvalues : array_like
        Array of p-values of shape (n_roi, n_times)

    References
    ----------
    Smith and Nichols, 2009 :cite:`smith2009threshold`
    """
    # get t-test over true and permuted mi
    t_obs, t_obs_surr = _rfx_ttest(mi, mi_p, center=center, zscore=zscore)
    # get (start, step) integration parameters
    if not isinstance(start, float):
        start = np.percentile(t_obs_surr, 100. * (1. - alpha))
    if not isinstance(step, float):
        stop = t_obs.max()
        step = (stop - start) / 100.
    th = dict(start=start, step=step)
    # infer p-values
    logger.info(f"    RFX non-parametric group t-test (alpha={alpha}, "
                f"threshold={th})")
    pvalues = temporal_clusters_permutation_test(t_obs, t_obs_surr, th, tail=1)

    return pvalues
