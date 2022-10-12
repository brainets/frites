"""Parametric statistics."""
import logging

import numpy as np
from scipy.stats import trim_mean


logger = logging.getLogger("frites")


def _trimmed(x, prop=.2, axis=0, keepdims=False):
    trm = trim_mean(x, prop, axis=axis)
    if keepdims:
        ax = [slice(None)] * x.ndim
        ax[axis] = np.newaxis
        trm = trm[tuple(ax)]
    return trm


RECENTER = {
    'mean': np.mean, 'median': np.median, 'zscore': np.mean,
    'trimmed': _trimmed
}


def _recenter(x, fcn_mean, zscore, axis=-1):
    """Recentering function."""
    _std = x.std(axis=axis, keepdims=True) if zscore else 1.
    return (x - fcn_mean(x, axis=axis, keepdims=True)) / _std


def ttest_1samp(x, pop_mean, axis=0, implementation='mne', sigma=0.001, **kw):
    """One-sample t-test.

    Parameters
    ----------
    x : array_like
        Sample observation
    pop_mean : float
        Expected value in the null hypothesis
    axis : int | 0
        Axis along which to perform the t-test
    implementation : {'mne', 'scipy'}
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
    if implementation == 'scipy':
        from scipy.stats import ttest_1samp as sp_ttest
        def fcn(x, pop_mean, axis):  # noqa
            return sp_ttest(x, pop_mean, axis=axis, **kw)[0]
    elif implementation == 'mne':
        from mne.stats import ttest_1samp_no_p as mne_ttest
        def fcn(x, pop_mean, axis):  # noqa
            return mne_ttest(np.moveaxis(x, axis, 0) - pop_mean, sigma=sigma,
                             **kw)

    return fcn(x, pop_mean, axis)


def rfx_ttest(mi, mi_p, center=False, sigma=0.001, ttested=False):
    """Perform the t-test across subjects.

    Parameters
    ----------
    mi : array_like
        A list of length n_roi of array of mutual information of shape
        (n_suj, n_times). If `ttested` is True, n_suj shoud be 1.
    mi_p : array_like
        A list of array of permuted mutual information of shape
        (n_perm, n_suj, n_times). If `ttested` is True, n_suj shoud be 1.
    sigma : float | 0.001
        Hat adjustment method, a value of 1e-3 may be a reasonable choice
    center : {False, 'mean', 'median', 'trimmed', 'zscore'}
        Re-center the time-series of effect arround 0 before computing the
        t-test. This parameters can be useful in case of a different number
        of data per brain region.
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
    zscore = center == 'zscore'
    if center in RECENTER.keys():
        # get the centering function
        fcn_mean = RECENTER[center]

        # here, we need to make a copy of the effect sizes to avoid changing
        # the ouputs
        mi, mi_p = mi.copy(), mi_p.copy()
        for k in range(n_roi):
            mi[k] = _recenter(mi[k], fcn_mean, zscore, axis=-1).copy()
            mi_p[k] = _recenter(mi_p[k], fcn_mean, zscore, axis=-1).copy()

    # get the mean of surrogates (low ram method)
    n_element = np.sum([np.prod(k.shape) for k in mi_p])
    sum_all = np.sum([np.sum(k) for k in mi_p])
    pop_mean_surr = sum_all / n_element

    """sigma estimation
    Here, the data are organized into a list of length (roi,), which means
    that the MNE t-test is going to evaluate one sigma per roi. To fix that,
    we estimate this sigma using the variance of all of the data
    """
    s_hat = sigma

    # sigma on true mi and permuted mi
    if s_hat > 0:
        sigma = s_hat * max([np.var(k, axis=0, ddof=1).max() for k in mi])
        sigma_p = s_hat * max([np.var(k, axis=1, ddof=1).max() for k in mi_p])
    else:
        sigma = sigma_p = 0.
    logger.debug(f"sigma_true={sigma}; sigma_permuted={sigma_p}")
    kw = dict(implementation='mne', method='absolute', sigma=sigma)
    kw_p = dict(implementation='mne', method='absolute', sigma=sigma_p)

    # perform the one sample t-test against the mean both on the true and
    # permuted mi
    logger.info(f"    T-test across subjects (pop_mean={pop_mean_surr}; "
                f"center={center}; zscore={zscore}; sigma={s_hat})")
    t_obs = np.stack([ttest_1samp(
        mi[k], pop_mean_surr, axis=0, **kw) for k in range(n_roi)])
    t_obs_surr = np.stack([ttest_1samp(
        mi_p[k], pop_mean_surr, axis=1, **kw_p) for k in range(n_roi)])
    t_obs_surr = np.swapaxes(t_obs_surr, 0, 1)

    return t_obs, t_obs_surr, pop_mean_surr
