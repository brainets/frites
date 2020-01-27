"""Functions for correction for multiple comparisons."""
import logging

import numpy as np
from mne.stats import fdr_correction, bonferroni_correction

logger = logging.getLogger('frites')


def permutation_mcp_correction(x, x_p, tail=1, mcp='maxstat', inplace=False):
    """Correction for MCP using non-parametric statistics.

    Parameters
    ----------
    x : array_like
        Array of true effect
    x_p : array_like
        Array of permutations of shape (..., n_perm) where the other dimensions
        should be the same as `x`
    tail : {-1, 0, 1}
        Type of comparison. Use -1 for the lower part of the distribution,
        1 for the higher part and 0 for both
    mcp : {'maxstat', 'fdr', 'bonferroni'}
        Method to use for correcting p-values for the multiple comparison
        problem. By default, maximum statistics is used
    inplace : bool | False
        Specify if operations can be performed inplace (faster and decrease
        ram usage but change the data)

    Returns
    -------
    pvalues : array_like
        Array of pvalues corrected for MCP with the same shape as the input `x`
    """
    assert tail in [-1, 0, 1]
    assert mcp in ['maxstat', 'fdr', 'bonferroni']
    assert isinstance(x, np.ndarray) and isinstance(x_p, np.ndarray)
    n_perm = x_p.shape[-1]

    logger.info(f"    Perform correction for MCP (mcp={mcp}; tail={tail})")

    # -------------------------------------------------------------------------
    # change the distribution according to the tail (support inplace operation)
    if tail == 1:     # upper part of the distribution
        pass
    elif tail == -1:  # bottom part of the distribution
        if inplace:
            x *= -1
            x_p *= -1
        else:
            x, x_p = -x, -x_p
    elif tail == 0:   # both part of the distribution
        if inplace:
            np.abs(x, out=x)
            np.abs(x_p, out=x_p)
        else:
            x, x_p = np.abs(x), np.abs(x_p)
    x = x[..., np.newaxis]

    # -------------------------------------------------------------------------
    # mcp correction
    if mcp is 'maxstat':
        # maximum over all dimensions except the perm one
        axis = tuple(np.arange(x_p.ndim - 1).tolist())
        x_p = np.max(x_p, keepdims=True, axis=axis)
        pv = (x <= x_p).sum(-1) / n_perm
        pv = np.clip(pv, 1. / n_perm, 1.)
    else:
        pv = (x <= x_p).sum(-1) / n_perm
        if mcp is 'fdr':
            pv = fdr_correction(pv, .05)[1]
        if mcp is 'bonferroni':
            pv = bonferroni_correction(pv, .05)[1]
    pv = np.clip(pv, 0., 1.)

    return pv
