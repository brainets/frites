"""Functions for correction for multiple comparisons."""
import logging

import numpy as np

from mne.stats.cluster_level import _find_clusters
from mne.stats import fdr_correction, bonferroni_correction

logger = logging.getLogger('frites')


###############################################################################
###############################################################################
#                         TESTWISE CORRECTION MCP
###############################################################################
###############################################################################


def testwise_correction_mcp(x, x_p, tail=1, mcp='maxstat'):
    """Test-wise correction for MCP using non-parametric statistics.

    This function can be used to correct the p-values for multiple comparisons
    at the test level (i.e at each time point, each roi, each frequencies
    etc.). This kind of correction usually suffers from a low statistical power
    (i.e if an effect is present, you might miss it because the correction is
    to conservative).

    Parameters
    ----------
    x : array_like
        Array of true effect
    x_p : array_like
        Array of permutations of shape (n_perm, ...) where the other dimensions
        should be the same as `x`
    tail : {-1, 0, 1}
        Type of comparison. Use -1 for the lower part of the distribution,
        1 for the higher part and 0 for both
    mcp : {'maxstat', 'fdr', 'bonferroni'}
        Method to use for correcting p-values for the multiple comparison
        problem. By default, maximum statistics is used

    Returns
    -------
    pvalues : array_like
        Array of pvalues corrected for MCP with the same shape as the input `x`
    """
    assert tail in [-1, 0, 1]
    assert mcp in ['maxstat', 'fdr', 'bonferroni']
    assert isinstance(x, np.ndarray) and isinstance(x_p, np.ndarray)
    n_perm = x_p.shape[0]

    logger.info(f"    Perform correction for MCP (mcp={mcp}; tail={tail})")

    # -------------------------------------------------------------------------
    # change the distribution according to the tail (support inplace operation)
    if tail == 1:     # upper part of the distribution
        pass
    elif tail == -1:  # bottom part of the distribution
        x, x_p = -x, -x_p
    elif tail == 0:   # both part of the distribution
        x, x_p = np.abs(x), np.abs(x_p)
    x = x[np.newaxis, ...]

    # -------------------------------------------------------------------------
    # mcp correction
    if mcp == 'maxstat':
        x_p_sh = tuple([n_perm] + [1] * (x.ndim - 1))
        # maximum over all dimensions except the perm one
        x_p = x_p.reshape(n_perm, -1).max(1).reshape(*x_p_sh)
        pv = (x <= x_p).sum(0) / n_perm
        pv = np.clip(pv, 1. / n_perm, 1.)
    else:
        pv = (x <= x_p).sum(0) / n_perm
        if mcp == 'fdr':
            pv = fdr_correction(pv, .05)[1]
        if mcp == 'bonferroni':
            pv = bonferroni_correction(pv, .05)[1]
    pv = np.clip(pv, 0., 1.)

    return pv


###############################################################################
###############################################################################
#                      CLUSTER-BASED CORRECTION MCP
###############################################################################
###############################################################################


def cluster_correction_mcp(x, x_p, th, tail=1, **kwargs):
    """Cluster-based correction for MCP using non-parametric statistics.

    This function can be used to compute the p-values, corrected for multiple
    comparisons using cluster-based. Note that this function detect clusters
    for each ROI but can be performed across multiple other dimensions (e.g
    frequencies, times etc.).

    Parameters
    ----------
    x : array_like
        Array of true effect size of shape (n_roi, ...)
    x_p : array_like
        Array of permutations of shape (n_perm, n_roi, ...)
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
    n_perm, n_roi = x_p.shape[0], x.shape[0]
    assert tail in [-1, 0, 1]
    kwargs['tail'] = tail

    logger.info(f"    Cluster detection (threshold={th}; tail={tail})")

    # -------------------------------------------------------------------------
    # identify clusters for the true x
    cl_loc, cl_mass = [], []
    for r in range(n_roi):
        _cl_loc, _cl_mass = _find_clusters(x[r, ...], th, **kwargs)
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
            _, __cl_p_null = _find_clusters(x_p[p, r, ...], th, **kwargs)
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
    cl_p_mass = np.tile(cl_p_mass.max(axis=0, keepdims=True), (n_roi, 1))
    pv = _clusters_to_pvalues(x.shape, n_perm, cl_loc, cl_mass, cl_p_mass)
    pv = np.clip(pv, 1. / n_perm, 1.)

    return pv


def _clusters_to_pvalues(x_shape, n_perm, cl_loc, cl_mass, cl_p_mass):
    """Transform clusters into p-values."""
    pvalues = np.full(x_shape, 1.)
    roi_free_shape = list(x_shape)[1::]
    for r, (cl_g, clm_g) in enumerate(zip(cl_loc, cl_mass)):
        for cl, clm in zip(cl_g, clm_g):
            pv = (clm <= cl_p_mass[[r], :]).sum(1) / n_perm
            if len(roi_free_shape) > 1:
                cl = cl.reshape(roi_free_shape)
            pvalues[r, cl] = pv
    return pvalues


def cluster_threshold(x, x_p, alpha=.05, tail=1, tfce=False, n_steps=100,
                      h_power=2, e_power=.5):
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
        Use Threshold Free Cluster Enhancement. Use either :

            * True to specify that TFCE have to be used. In that case, the
              start and integration step are automatically inferred
            * A dict that contains a 'start' and a 'step' key to manually set
              TFCE parameters. Could also use 'e_power' and 'h_power' entries
            * A dict that is only there to set TFCE parameters n_steps, e_power
              and h_power but using the automatic starting and integration step
    n_steps : int | 100
        Number of integration steps between the start and stoping values for
        the TFCE
    h_power : float | 2
        Default H exponent of the TFCE
    e_power : float | .5
        Default E exponent of the TFCE

    Returns
    -------
    threshold : float, dict
        The cluster threshold. For the TFCE, this output a is dictionary with
        a 'start' and a 'step' keys (MNE convention)
    """
    logger.info(f"    Cluster forming threshold (tail={tail}; alpha={alpha}; "
                f"tfce={tfce})")
    kw = dict(interpolation='nearest')
    if tfce or isinstance(tfce, dict):
        # tfce is a dict that is only used to set MNE parameters
        # (n_steps, e_power, h_power)
        if isinstance(tfce, dict):
            n_steps = tfce.get('n_steps', n_steps)
            e_power = tfce.get('e_power', e_power)
            h_power = tfce.get('h_power', h_power)
            # now that we have the parameters, reset the tfce to use auto
            # in case 'start' and 'step' keys are absents
            if ('start' not in tfce.keys()) and ('step' not in tfce.keys()):
                tfce = True
        if not isinstance(tfce, dict):
            if tail == 1:
                start = max(np.nanpercentile(x_p, 100 * (1 - alpha), **kw), 0.)
                stop = x.max()
            elif tail == -1:
                start = min(np.nanpercentile(x_p, 100. * alpha, **kw), 0.)
                stop = x.min()
            elif tail == 0:
                start = np.nanpercentile(np.abs(x_p), 100 * (1 - alpha), **kw)
                stop = np.abs(x).max()
            step = (stop - start) / n_steps
            th = dict(start=start, step=step)
        else:
            th = tfce
        th.update(dict(e_power=e_power, h_power=h_power))
        assert all([k in th.keys() for k in ['start', 'step']])
    else:
        if tail == 1:
            th = np.nanpercentile(x_p, 100. * (1. - alpha), **kw)
        elif tail == -1:
            th = np.nanpercentile(x_p, 100. * alpha, **kw)
        elif tail == 0:
            th = np.nanpercentile(np.abs(x_p), 100. * (1. - alpha), **kw)

    return th
