"""Statistical workflow for electrophysiological data."""
import logging

import numpy as np

from frites.stats import STAT_FUN
from frites.io import set_log_level

logger = logging.getLogger("frites")


class WfStatsEphy(object):
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
        set_log_level(verbose)
        logger.info("Definition of a non-parametric statistical workflow")

    def fit(self, effect, perms, stat_method="rfx_cluster_ttest", **kw_stats):
        """Fit the workflow on true data.

        Parameters
        ----------
        effect : list
            True effect list of length (n_roi,) composed of arrays each one of
            shape (n_subjects, n_times). Number of subjects per ROI could be
            different
        perms : list
            Permutation list of length (n_roi,) composed of arrays each one of
            shape (n_perm, n_subjects, n_times). Number of subjects per ROI
            could be different
        stat_method : string | "rfx_cluster_ttest"
            Statistical method to use. Method names depends on the initial
            choice of inference type (ffx=fixed effect or rfx=random effect).

            **For the fixed effect (ffx) :**

                * 'ffx_maxstat' : maximum statistics correction (see
                  :func:`frites.stats.ffx_maxstat`)
                * 'ffx_fdr' : False Discovery Rate correction (see
                  :func:`frites.stats.ffx_fdr`)
                * 'ffx_bonferroni' : Bonferroni correction (see
                  :func:`frites.stats.ffx_bonferroni`)
                * 'ffx_cluster_maxstat' : maximum statistics correction at
                  cluster level (see :func:`frites.stats.ffx_cluster_maxstat`)
                * 'ffx_cluster_fdr' : False Discovery Rate correction at
                  cluster level (see :func:`frites.stats.ffx_cluster_fdr`)
                * 'ffx_cluster_bonferroni' : Bonferroni correction at
                  cluster level (see
                  :func:`frites.stats.ffx_cluster_bonferroni`)
                * 'ffx_cluster_tfce' : Threshold Free Cluster Enhancement for
                  cluster level inference (see
                  :func:`frites.stats.ffx_cluster_tfce`
                  :cite:`smith2009threshold`)

            **For the random effect (rfx) :**

                * 'rfx_cluster_ttest' : t-test across subjects for cluster
                  level inference (see :func:`frites.stats.rfx_cluster_ttest`)
                * 'rfx_cluster_ttest_tfce' : t-test across subjects combined
                  with the Threshold Free Cluster Enhancement for cluster level
                  inference (see :func:`frites.stats.rfx_cluster_ttest_tfce`
                  :cite:`smith2009threshold`)
        output_type : {'array', 'dataframe', 'dataarray'}
            Convert the mutual information and p-values arrays either to
            pandas DataFrames (require pandas to be installed) either to a
            xarray DataArray or DataSet (require xarray to be installed)
        kw_stats : dict | {}
            Additional arguments to pass to the selected statistical method
            selected using the `stat_method` input parameter

        Returns
        -------
        pvalues : array_like
            Array of p-values of shape (n_times, n_roi)
        tvalues : array_like
            Array of t-values of shape (n_times, n_n_roi). This ouput is only
            computed for group-level analysis
        """
        # ---------------------------------------------------------------------
        # check inputs
        # ---------------------------------------------------------------------
        assert isinstance(effect, list) and isinstance(perms, list)
        assert all([isinstance(k, np.ndarray) and k.ndim == 2 for k in effect])
        n_roi, n_times, tvalues = len(effect), effect[0].shape[1], None
        # don't compute statistics if stat_method is None
        if (stat_method is None) or not len(perms):
            return np.ones((n_times, n_roi), dtype=float), tvalues
        assert all([isinstance(k, np.ndarray) and k.ndim == 3 for k in perms])
        assert len(effect) == len(perms)

        # ---------------------------------------------------------------------
        # check that the selected method does exist
        # ---------------------------------------------------------------------
        inference = stat_method[0:3]
        if stat_method not in STAT_FUN[inference].keys():
            m_names = [k.__name__ for k in STAT_FUN[inference].values()]
            raise KeyError(f"Selected statistical method `{stat_method}` "
                           f"doesn't exist. For {inference} inference, "
                           f"use either : {', '.join(m_names)}")
        stat_fun = STAT_FUN[inference][stat_method]

        # ---------------------------------------------------------------------
        # compute stats according to inference type
        # ---------------------------------------------------------------------
        level = 'cluster' if 'cluster' in stat_fun.__name__ else "time-point"
        logger.info(f"    Run the statistical workflow ({stat_method}) for "
                    f"inferences at the {level}-level")
        if inference == 'ffx':
            # for the fixed effect, since it's computed across subjects it
            # means that each roi has an mi.shape of (1, n_times) and can then
            # been concatenated over the first axis. Same for the permutations,
            # with a shape of (n_perm, 1, n_times)
            effect = np.concatenate(effect, axis=0)
            perms = np.concatenate(perms, axis=1)
            # compute the p-values
            pvalues = stat_fun(effect, perms, **kw_stats)
        elif inference == 'rfx':
            # for random effect, the number of subjects can't be bellow 2
            # other wise the t-test is going to failed
            n_suj_roi = [k.shape[0] for k in effect]
            n_suj_min, n_suj_argmin = np.min(n_suj_roi), np.argmin(n_suj_roi)
            if n_suj_min < 2:
                raise ValueError(
                    f"The number of subjects of roi {n_suj_argmin} has "
                    f"{n_suj_min} subjects. The minimum number of subjects for"
                    " random effect should not be under 2.")
            # compute p-values and t-values
            pvalues, tvalues = stat_fun(effect, perms, **kw_stats)

        # ---------------------------------------------------------------------
        # postprocessing
        # ---------------------------------------------------------------------
        # by default p and t-values are (n_roi, n_times)
        if isinstance(tvalues, np.ndarray):
            tvalues = tvalues.T
        pvalues = pvalues.T

        return pvalues, tvalues
