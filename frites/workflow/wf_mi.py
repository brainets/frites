"""Workflow for computing MI and evaluate statistics."""
import logging

import numpy as np
from joblib import Parallel, delayed

from frites.core import MI_FUN, permute_mi_vector
from frites.stats import STAT_FUN
from frites.config import CONFIG

logger = logging.getLogger("frites")


class WorkflowMiStats(object):
    """Workflow of mutual-information and statistics.

    This class allows to define a workflow for computing the mutual information
    and then to evaluate the significance using non-parametric statistics
    (either within-subjects or between subjects).

    Parameters
    ----------
    mi_type : {'cc', 'cd', 'ccd'}
        The type of mutual information that is going to be performed. Use
        either :

            * 'cc' : mutual information between two continuous variables
            * 'cd' : mutual information between a continuous and a discret
              variables
            * 'ccd' : mutual information between two continuous variables
              conditioned by a third discret one
    inference : {"ffx", "rfx"}
        Statistical inferences that is desired. Use either :

            * 'ffx' : fixed-effect to make inferences only for the population
              that have been used
            * 'rfx' : random-effect to generalize inferences to a random
              population.

        By default, the workflow uses group level inference ('rfx')

    References
    ----------
    Friston et al., 1996, 1999 :cite:`friston1996detecting,friston1999many`
    """

    def __init__(self, mi_type='cc', inference='rfx'):
        """Init."""
        assert mi_type in ['cc', 'cd', 'ccd']
        assert inference in ['ffx', 'rfx']
        self._mi_type = mi_type
        self._inference = inference

        logger.info(f"Workflow for computing mutual information ({mi_type}) and"
                    f" statistics ({inference}) has been defined")

    ###########################################################################
    #                                INTERNALS
    """
        1 - Prepare the data (group by roi and copnorm)
        2 - Compute the MI (mi) and the permuted MI (mi_p)
        3 - Evaluate the p-values using non-parametric statistics
        4 - Post-process outputs
    """
    ###########################################################################

    def _node_prepare_data(self, dataset):
        """Prepare the data before computing the mi.

        The preparation of the data depends both on the type of inferences
        that are required and on the type of mutual information that is going
        to be performed.
        """
        # inplace preparation
        dataset.groupby("roi")
        dataset.copnorm(mi_type=self._mi_type, inference=self._inference)


    def _node_compute_mi(self, dataset, n_perm=1000, n_jobs=-1):
        """Compute mi and permuted mi.

        Permutations are performed by randomizing the regressor variable. For
        the fixed effect, this randomization is performed across subjects. For
        the random effect, the randomization is performed per subject.
        """
        # get the function for computing mi
        mi_fun = MI_FUN[self._mi_type][self._inference]
        assert self._inference in mi_fun.__name__
        assert f"_{CONFIG['COPULA_CONV'][self._mi_type]}_" in mi_fun.__name__
        # get x, y, z and subject names per roi
        x, y, z, suj = dataset.x, dataset.y, dataset.z, dataset.suj_roi
        n_roi = dataset.n_roi
        # evaluate true mi
        logger.info(f"    Evaluate true and permuted mi (n_perm={n_perm}, "
                    f"n_jobs={n_jobs})")
        mi = [mi_fun(x[k], y[k], z[k], suj[k]) for k in range(n_roi)]
        # evaluate permuted mi
        mi_p = []
        for r in range(n_roi):
            # get the randomize version of y
            y_p = permute_mi_vector(y[r], suj[r], mi_type=self._mi_type,
                                    inference=self._inference, n_perm=n_perm)
            # run permutations using the randomize regressor
            _mi = Parallel(n_jobs=n_jobs, **CONFIG["JOBLIB_CFG"])(delayed(
               mi_fun)(x[r], y_p[p], z[r], suj[r]) for p in range(n_perm))
            mi_p += [np.asarray(_mi)]

        return mi, mi_p


    def _node_compute_stats(self, mi, mi_p, n_jobs=-1,
                            stat_method='rfx_cluster_ttest', **kw_stats):
        """Compute the non-parametric statistics.

        mi   = list of length n_roi composed with arrays of shape
               (n_subjects, n_times)
        mi_p = list of length n_roi composed with arrays of shape
               (n_perm, n_subjects, n_times)
        """
        # get the function to evaluate statistics
        stat_fun = STAT_FUN[f"{stat_method}"]
        assert self._inference in stat_fun.__name__, (
            f"the select function is not compatible with {self._inference} "
            "inferences")
        # concatenate mi (if needed)
        if self._inference == 'ffx':
            # for the fixed effect, since it's computed across subjects it
            # means that each roi has an mi.shape of (1, n_times) and can then
            # been concatenated over the first axis. Same for the permutations,
            # with a shape of (n_perm, 1, n_times)
            mi, mi_p = np.concatenate(mi, axis=0), np.concatenate(mi_p, axis=1)
            # get the p-values
            pvalues = stat_fun(mi, mi_p, **kw_stats)
        elif self._inference == 'rfx':
            pvalues = stat_fun(mi, mi_p, **kw_stats)

        return pvalues

    def _node_postprocessing(self, mi, pvalues, mean_mi=True):
        """Post preocess outputs."""
        if mean_mi:
            logger.info("    Mean mi across subjects")
            mi = np.stack([k.mean(axis=0) for k in mi])
        return mi, pvalues


    ###########################################################################
    #                             EXTERNALS
    ###########################################################################

    def fit(self, dataset, n_perm=1000, n_jobs=-1,
            stat_method='rfx_cluster_ttest', **kw_stats):
        """Run the workflow on a dataset.

        In order to run the worflow, you must first provide a dataset instance
        (see :class:`frites.dataset.DatasetEphy`)

        .. warning::

            When performing statistics at the cluster-level, we only test
            the cluster size. This means that in your results, you can only
            discuss about the presence of a significant cluster without being
            precise about its spatio-temporal properties
            (see :cite:`sassenhagen2019cluster`)

        Parameters
        ----------
        dataset : :class:`frites.dataset.DatasetEphy`
            A dataset instance
        n_perm : int | 1000
            Number of permutations to perform in order to estimate the random
            distribution of mi that can be obtained by chance
        n_jobs : int | -1
            Number of jobs to use for parallel computing (use -1 to use all
            jobs)
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
                  with the TFCE for cluster level inference (see
                  :func:`frites.stats.rfx_cluster_ttest_tfce`
                  :cite:`smith2009threshold`)
        kw_stats : dict | {}
            Additional arguments to pass to the selected statistical method
            selected using the `stat_method` input parameter

        Returns
        -------
        pvalues : array_like
            Array of p-values of shape (n_roi, n_times)

        References
        ----------
        Maris and Oostenveld, 2007 :cite:`maris2007nonparametric`
        """
        self._node_prepare_data(dataset)
        mi, mi_p = self._node_compute_mi(dataset, n_perm=n_perm, n_jobs=n_jobs)
        pvalues = self._node_compute_stats(mi, mi_p, n_jobs=n_jobs,
                                           stat_method=stat_method, **kw_stats)
        mi, pvalues = self._node_postprocessing(mi, pvalues, mean_mi=True)

        return mi, pvalues


    def convert_outputs(self, mi, pvalues, convert_to='dataframe'):
        """Convert outputs.

        Parameters
        ----------
        mi : array_like
            Array of mutual information of shape (n_roi, n_times)
        pvalues : array_like
            Array of p-values of shape (n_roi, n_times)
        convert_to : {'dataframe', 'dataarray', 'dataset'}
            Convert the mutual information and p-values arrays either to
            pandas DataFrames (require pandas to be installed) either to a
            xarray DataArray (require xarray to be installed)
        """
        # 'dataframe', 'dataarray', 'dataset' <-
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from frites.simulations import sim_multi_suj_ephy, sim_mi_cc
    from frites.dataset import DatasetEphy
    from frites.core import mi_nd_gg

    modality = 'meeg'
    n_subjects = 3
    n_epochs = 100
    n_times = 40
    n_roi = 3
    n_sites_per_roi = 1
    as_mne = False
    x, roi, time = sim_multi_suj_ephy(n_subjects=n_subjects, n_epochs=n_epochs,
                                      n_times=n_times, n_roi=n_roi,
                                      n_sites_per_roi=n_sites_per_roi,
                                      as_mne=as_mne, modality=modality,
                                      random_state=1)
    y, _ = sim_mi_cc(x, snr=.8)
    time = np.arange(n_times) / 512

    # y_t = np.tile(y[0].reshape(-1, 1, 1), (1, x[0].shape[1], 100))
    # print(x[0].shape, y[0].shape)
    # mi_t = mi_nd_gg(x[0], y_t, traxis=0)

    # plt.plot(mi_t.T)
    # plt.show()
    # exit()

    dt = DatasetEphy(x, y, roi=roi, times=time)
    wf = WorkflowMiStats('cc', 'rfx')
    mi, pvalues = wf.fit(dt, n_jobs=-1, n_perm=20,
                         stat_method='rfx_cluster_ttest_tfce',
                         center='median', zscore=False)

    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.plot(time, mi.T)
    plt.subplot(212)
    plt.plot(time, pvalues.T)

    plt.show()
