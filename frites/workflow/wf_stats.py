"""Statistical workflow for electrophysiological data."""
import numpy as np

from frites.stats.stats_mcp import (testwise_correction_mcp,
                                    cluster_correction_mcp, cluster_threshold)
from frites.stats.stats_param import rfx_ttest

from frites.workflow.wf_base import WfBase
from frites.io import set_log_level, logger


class WfStats(WfBase):
    """Workflow of non-parametric statistics for electropĥysiological data.

    The goal of this workflow is to provide an interface for assessing non-
    parametric statistics on electrophysiological data based on anatomical
    informations (ROI = Region Of Interest). In particular, it allows to make
    inferences either about a specific population (FFX = fixed effect) or build
    a model of a random population (RFX = Random effect). In addition,
    significant effect can either be detected at the spatio-temporal level
    (i.e at each time point and for each ROI) either at the cluster-level (i.e
    temporal clusters).
    """

    def __init__(self, verbose=None):  # noqa
        WfBase.__init__(self)
        set_log_level(verbose)
        logger.info("Definition of a non-parametric statistical workflow")

    def fit(self, effect, perms, inference='rfx', mcp='cluster', tail=1,
            cluster_th=None, cluster_alpha=0.05, ttested=False):
        """Fit the workflow on true data.

        Parameters
        ----------
        effect : list
            True effect list of length (n_roi,) composed of arrays each one of
            shape (n_subjects, ..., n_times). Number of subjects per ROI could
            be different
        perms : list
            Permutation list of length (n_roi,) composed of arrays each one of
            shape (n_perm, n_subjects, ..., n_times). Number of subjects per
            ROI could be different
        inference : {'ffx', 'rfx'}
            Perform either Fixed-effect ('ffx') or Random-effect ('rfx')
            inferences. By default, random-effect is used
        mcp : {'cluster', 'maxstat', 'fdr', 'bonferroni', 'nostat', None}
            Method to use for correcting p-values for the multiple comparison
            problem. Use either :

                * 'cluster' : cluster-based statistics [default]
                * 'maxstat' : test-wise maximum statistics correction
                * 'fdr' : test-wise FDR correction
                * 'bonferroni' : test-wise Bonferroni correction
                * 'nostat' : permutations are computed but no statistics are
                  performed
                * 'noperm' / None : no permutations are computed
        tail : {-1, 0, 1}
            Type of comparison. Use -1 for the lower part of the distribution,
            1 for the higher part and 0 for both. By default, upper tail of the
            distribution is used
        cluster_th : str, float | None
            The threshold to use for forming clusters. Use either :

                * a float that is going to act as a threshold
                * None and the threshold is automatically going to be inferred
                  using the distribution of permutations
                * 'tfce' : for Threshold Free Cluster Enhancement
        cluster_alpha : float | 0.05
            Control the percentile to use for forming the clusters. By default
            the 95th percentile of the permutations is used.
        ttested : bool | False
            Specify if the inputs have already been t-tested

        Returns
        -------
        pvalues : array_like
            Array of p-values of shape (..., n_times, n_roi)
        tvalues : array_like
            Array of t-values of shape (..., n_times, n_roi). This ouput is
            only computed for group-level analysis

        References
        ----------
        Smith and Nichols, 2009 :cite:`smith2009threshold`
        """
        # ---------------------------------------------------------------------
        # check inputs
        # ---------------------------------------------------------------------
        assert inference in ['ffx', 'rfx']
        assert mcp in ['cluster', 'maxstat', 'fdr', 'bonferroni', 'nostat',
                       'noperm', None]
        assert isinstance(effect, list) and isinstance(perms, list)
        assert all([isinstance(k, np.ndarray) and k.ndim >= 2 for k in effect])
        n_roi, n_times, tvalues = len(effect), effect[0].shape[-1], None
        # don't compute statistics if `mcp` is None
        if (mcp in [None, 'noperm']) or not len(perms):
            return np.ones((n_times, n_roi), dtype=float), tvalues
        assert all([isinstance(k, np.ndarray) and k.ndim >= 3 for k in perms])
        assert len(effect) == len(perms)
        # test that all values are finite
        assert all([np.isfinite(k).all() for k in effect])
        assert all([np.isfinite(k).all() for k in perms])

        # ---------------------------------------------------------------------
        # FFX / RFX
        # ---------------------------------------------------------------------
        nb_suj_roi = [k.shape[0] for k in effect]
        if inference == 'ffx':
            # check that the number of subjects is 1
            ffx_suj = np.max(nb_suj_roi) == 1
            assert ffx_suj, "For FFX, `n_subjects` should be 1"
            es, es_p = effect, perms
            logger.info("    Fixed-effect inference (FFX)")
            # es = (n_roi, n_times); es_p = (n_perm, n_roi, n_times)
            es, es_p = np.concatenate(es, axis=0), np.concatenate(es_p, axis=1)
        elif inference == 'rfx':
            if ttested:
                es = np.concatenate(effect, axis=0)
                es_p = np.concatenate(perms, axis=1)
            else:
                # check that the number of subjects is > 1
                rfx_suj = np.min(nb_suj_roi) > 1
                assert rfx_suj, "For RFX, `n_subjects` should be > 1"
                # modelise how subjects are distributed
                es, es_p, pop_mean = rfx_ttest(effect, perms)
                from frites.config import CONFIG
                sigma = CONFIG['TTEST_MNE_SIGMA']
                self.attrs.update(dict(ttest_pop_mean=pop_mean,
                                       ttest_sigma=sigma))
            tvalues = es

        # ---------------------------------------------------------------------
        # cluster forming threshold
        # ---------------------------------------------------------------------
        if mcp == 'cluster':
            if isinstance(cluster_th, (int, float)):
                th, tfce = cluster_th, None
            else:
                if (cluster_th == 'tfce'):          # TFCE auto
                    tfce = True
                elif isinstance(cluster_th, dict):  # TFCE manual
                    tfce = cluster_th
                else:
                    tfce = None                     # cluster_th is None
                th = cluster_threshold(es, es_p, alpha=cluster_alpha,
                                       tail=tail, tfce=tfce)
                self._cluster_th = cluster_th
            self.attrs.update(dict(th=th, tfce=tfce))

        # ---------------------------------------------------------------------
        # test-wise or cluster-based correction for multiple comparisons
        # ---------------------------------------------------------------------
        if mcp == 'cluster':
            logger.info('    Inference at cluster-level')
            pvalues = cluster_correction_mcp(es, es_p, th, tail=tail)
        else:
            logger.info('    Inference at spatio-temporal level (test-wise)')
            pvalues = testwise_correction_mcp(es, es_p, tail=tail, mcp=mcp)

        # ---------------------------------------------------------------------
        # postprocessing
        # ---------------------------------------------------------------------
        # by default p and t-values are (n_roi, n_times)
        if isinstance(tvalues, np.ndarray):
            tvalues = np.moveaxis(tvalues, 0, -1)
        pvalues = np.moveaxis(pvalues, 0, -1)

        # update internal config
        self.attrs.update(dict(
            inference=inference, mcp=mcp, tail=tail, cluster_th=cluster_th,
            cluster_alpha=cluster_alpha, ttested=int(ttested)))
        return pvalues, tvalues

    @property
    def cluster_th(self):
        """Cluster forming threshold."""
        return self._cluster_th
