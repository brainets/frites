"""Parametric statistics."""
import numpy as np


def ttest_1samp(x, pop_mean, axis=0, method='mne'):
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

    Returns
    -------
    tvalues : array_like
        Array of t-values
    """
    if method == 'scipy':
        from scipy.stats import ttest_1samp as sp_ttest
        def fcn(x, pop_mean, axis):  # noqa
            return sp_ttest(x, pop_mean, axis=axis)[0]
    elif method == 'mne':
        from mne.stats import ttest_1samp_no_p as mne_ttest
        def fcn(x, pop_mean, axis):  # noqa
            return mne_ttest(np.moveaxis(x, axis, 0) - pop_mean, sigma=1e-3)

    return fcn(x, pop_mean, axis)
