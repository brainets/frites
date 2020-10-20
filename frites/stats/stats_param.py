"""Parametric statistics."""
import logging

import numpy as np
from scipy.stats import trim_mean


logger = logging.getLogger("frites")


RECENTER = dict(mean=np.mean, median=np.median,
                trimmed=lambda x, axis=0: trim_mean(x, .2, axis=axis))


def ttest_1samp(x, pop_mean, axis=0, method='mne', sigma=0.001):
    """One-sample t-test.

    Parameters
    ----------
    x : array_like
        Sample observation
    pop_mean : float
        Expected value in the null hypothesis
    axis : int | 0
        Axis along which to perform the t-test
    method : {'mne', 'scipy'}
        Use either the scipy or the mne t-test
    sigma : float | 0.001
        Hat adjustment method, a value of 1e-3 may be a reasonable choice

    Returns
    -------
    tvalues : array_like
        Array of t-values

    References
    ----------
    Ridgway et al., 2012 :cite:`ridgway2012problem`
    """
    if method == 'scipy':
        from scipy.stats import ttest_1samp as sp_ttest
        def fcn(x, pop_mean, axis):  # noqa
            return sp_ttest(x, pop_mean, axis=axis)[0]
    elif method == 'mne':
        from mne.stats import ttest_1samp_no_p as mne_ttest
        def fcn(x, pop_mean, axis):  # noqa
            return mne_ttest(np.moveaxis(x, axis, 0) - pop_mean, sigma=sigma)

    return fcn(x, pop_mean, axis)


def rfx_ttest(mi, mi_p, center=False, zscore=False, ttested=False):
    """Perform the t-test across subjects.

    Parameters
    ----------
    mi : array_like
        A list of length n_roi of array of mutual information of shape
        (n_suj, n_times). If `ttested` is True, n_suj shoud be 1.
    mi_p : array_like
        A list of array of permuted mutual information of shape
        (n_perm, n_suj, n_times). If `ttested` is True, n_suj shoud be 1.
    center : {'mean', 'median', 'trimmed'} | False
        If True, substract the mean of the surrogates to the true and permuted
        mi. The median or the 20% trimmed mean can also be removed
        :cite:`wilcox2018guide`
    zscore : bool | False
        Apply z-score normalization to the true and permuted mutual information
    ttested : bool | False
        Specify if the inputs have already been t-tested

    Returns
    -------
    t_obs : array_like
        Array of true t-values of shape (n_suj, n_times)
    tobs_surr : array_like
        Array of permuted t-values of shape (n_perm, n_suj, n_times)
    pop_mean : float
        The value that have been used to compute the one-sample t-test. If the
        data have already been t-tested, this parameter is set to NaN

    References
    ----------
    Giordano et al., 2017 :cite:`giordano2017contributions`
    """
    # if data have already been t-tested, just return it
    if ttested:
        logger.debug("    Data already t-tested")
        t_obs = np.concatenate(mi, axis=0)
        t_obs_surr = np.concatenate(mi_p, axis=1)
        return t_obs, t_obs_surr, np.nan

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
    from frites.config import CONFIG
    kw = dict(method='mne', sigma=CONFIG['TTEST_MNE_SIGMA'])
    logger.info(f"    T-test across subjects (pop_mean={pop_mean_surr}; "
                f"center={center}; zscore={zscore}; sigma={kw['sigma']})")
    t_obs = np.stack([ttest_1samp(
        mi[k], pop_mean_surr, axis=0, **kw) for k in range(n_roi)])
    t_obs_surr = np.stack([ttest_1samp(
        mi_p[k], pop_mean_surr, axis=1, **kw) for k in range(n_roi)])
    t_obs_surr = np.swapaxes(t_obs_surr, 0, 1)

    return t_obs, t_obs_surr, pop_mean_surr
