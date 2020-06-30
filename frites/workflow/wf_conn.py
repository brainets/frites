"""Workflow of connectivity."""
import numpy as np
from joblib import Parallel, delayed

from frites import config
from frites.io import (set_log_level, logger, convert_dfc_outputs)
from frites.core import get_core_mi_fun, permute_mi_trials
from frites.workflow.wf_stats_ephy import WfStatsEphy
from frites.workflow.wf_base import WfBase


class WfConn(WfBase):
    """Workflow of instantaneous pairwise connectivity and statistics.

    This class allows to define a workflow for computing the instantaneous
    pairwise connectivity using mutual information and then to evaluate the
    significance with non-parametric statistics (either within-subjects or
    between subjects). Note that the MI is computed at each time point and
    across trials.

    Parameters
    ----------
    inference : {"ffx", "rfx"}
        Statistical inferences that is desired. Use either :

            * 'ffx' : fixed-effect to make inferences only for the population
              that have been used
            * 'rfx' : random-effect to generalize inferences to a random
              population.

        By default, the workflow uses group level inference ('rfx')
    mi_method : {'gc', 'bin'}
        Method for computing the mutual information. Use either :

            * 'gc' : gaussian-copula based mutual information. This is the
              fastest method but it can only captures monotonic relationships
              between variables
            * 'bin' : binning-based method that can captures any kind of
              relationships but is much slower and also required to define the
              number of bins to use. Note that if the Numba package is
              installed computations should be much faster
    kernel : array_like | None
        Kernel for smoothing true and permuted MI. For example, use
        np.hanning(3) for a 3 time points smoothing or np.ones((3)) for a
        moving average

    References
    ----------
    Friston et al., 1996, 1999 :cite:`friston1996detecting,friston1999many`
    """

    def __init__(self, inference='rfx', mi_method='gc', kernel=None,
                 verbose=None):
        """Init."""
        WfBase.__init__(self)
        assert inference in ['ffx', 'rfx'], (
            "'inference' input parameter should either be 'ffx' or 'rfx'")
        assert mi_method in ['gc', 'bin'], (
            "'mi_method' input parameter should either be 'gc' or 'bin'")
        self._mi_type = 'cc'
        self._inference = inference
        self._mi_method = mi_method
        self._need_copnorm = mi_method == 'gc'
        self._gcrn = inference is 'rfx'
        self._kernel = kernel
        set_log_level(verbose)
        self.clean()
        self._wf_stats = WfStatsEphy(verbose=verbose)
        # update internal config
        self.update_cfg(mi_type=self._mi_type, inference=inference,
            mi_method=mi_method, kernel=kernel)

        logger.info(f"Workflow for computing connectivity ({self._mi_type} - "
                    f"{mi_method})")

    def _node_prepare_data(self, dataset):
        """Prepare the data before computing the mi.

        The preparation of the data depends both on the type of inferences
        that are required and on the type of mutual information that is going
        to be performed.
        """
        # inplace preparation
        dataset.groupby("roi")
        if self._need_copnorm:
            dataset.copnorm(mi_type=self._mi_type, gcrn_per_suj=self._gcrn)
        # track time and roi
        self._times, self._roi = dataset.times, dataset.roi_names

    def _node_compute_mi(self, dataset, n_bins, n_perm, n_jobs, random_state):
        """Compute mi and permuted mi.

        Permutations are performed by randomizing the target roi. For the fixed
        effect, this randomization is performed across subjects. For the random
        effect, the randomization is performed per subject.
        """
        # get the function for computing mi
        mi_fun = get_core_mi_fun(self._mi_method)[f"{self._mi_type}_conn"]
        assert (f"mi_{self._mi_method}_ephy_conn_"
                f"{self._mi_type}" == mi_fun.__name__)
        # get x, y, z and subject names per roi
        x, y, suj = dataset.x, dataset.y, dataset.suj_roi
        roi = dataset.roi_names
        n_roi, inf = dataset.n_roi, self._inference
        # get the pairs for computing mi
        self.pairs = dataset.get_connectivity_pairs(
            nb_min_suj=dataset.nb_min_suj, directed=False)
        x_s, x_t = self.pairs
        n_pairs = len(self.pairs)
        # evaluate true mi
        logger.info(f"    Evaluate true and permuted mi (n_perm={n_perm}, "
                    f"n_jobs={n_jobs}, n_pairs={len(x_s)})")
        mi = [mi_fun(x[s], x[t], suj[s], suj[t], inf,
                     n_bins=n_bins) for s, t in zip(x_s, x_t)]
        # get joblib configuration
        cfg_jobs = config.CONFIG["JOBLIB_CFG"]
        # evaluate permuted mi
        mi_p = []
        for s, t in zip(x_s, x_t):
            # get the randomize version of y
            y_p = permute_mi_trials(suj[t], inference=self._inference,
                                    n_perm=n_perm)
            # run permutations using the randomize regressor
            _mi = Parallel(n_jobs=n_jobs, **cfg_jobs)(delayed(mi_fun)(
                x[s], x[t][..., y_p[p]], suj[s], suj[t], inf,
                n_bins=n_bins) for p in range(n_perm))
            mi_p += [np.asarray(_mi)]
        # # smoothing
        if isinstance(self._kernel, np.ndarray):
            logger.info("    Apply smoothing to the true and permuted MI")
            for r in range(len(mi)):
                for s in range(mi[r].shape[0]):
                    mi[r][s, :] = np.convolve(
                        mi[r][s, :], self._kernel, mode='same')
                    for p in range(mi_p[r].shape[0]):
                        mi_p[r][p, s, :] = np.convolve(
                            mi_p[r][p, s, :], self._kernel, mode='same')

        self._mi, self._mi_p = mi, mi_p

        return mi, mi_p

    def fit(self, dataset, mcp='cluster', n_perm=1000, cluster_th=None,
            cluster_alpha=0.05, n_bins=None, n_jobs=-1, random_state=None,
            **kw_stats):
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
        mcp : {'cluster', 'maxstat', 'fdr', 'bonferroni', 'nostat', None}
            Method to use for correcting p-values for the multiple comparison
            problem. Use either :
                
                * 'cluster' : cluster-based statistics [default]
                * 'maxstat' : test-wise maximum statistics correction
                * 'fdr' : test-wise FDR correction
                * 'bonferroni' : test-wise Bonferroni correction
                * 'noperm' / None : no permutations are computed
        n_perm : int | 1000
            Number of permutations to perform in order to estimate the random
            distribution of mi that can be obtained by chance
        cluster_th : str, float | None
            The threshold to use for forming clusters. Use either :

                * a float that is going to act as a threshold
                * None and the threshold is automatically going to be inferred
                  using the distribution of permutations
                * 'tfce' : for Threshold Free Cluster Enhancement
        cluster_alpha : float | 0.05
            Control the percentile to use for forming the clusters. By default
            the 95th percentile of the permutations is used.
        n_bins : int | None
            Number of bins to use if the method for computing the mutual
            information is based on binning (mi_method='bin'). If None, the
            number of bins is going to be automatically inferred based on the
            number of trials and variables
        n_jobs : int | -1
            Number of jobs to use for parallel computing (use -1 to use all
            jobs)
        random_state : int | None
            Fix the random state of the machine (use it for reproducibility).
            If None, a random state is randomly assigned.
        kw_stats : dict | {}
            Additional arguments to pass to the selected statistical method
            selected using the `stat_method` input parameter

        Returns
        -------
        mi, pvalues : array_like
            DataArray of mean mutual information and p-values.

        References
        ----------
        Maris and Oostenveld, 2007 :cite:`maris2007nonparametric`
        """
        # ---------------------------------------------------------------------
        # prepare variables
        # ---------------------------------------------------------------------
        # don't compute permutations if mcp is either nostat / None
        if mcp in ['noperm', None]:
            n_perm = 0
        # infer the number of bins if needed
        if (self._mi_method is 'bin') and not isinstance(n_bins, int):
            n_bins = 4
            logger.info(f"    Use an automatic number of bins of {n_bins}")
        self._n_bins = n_bins

        # ---------------------------------------------------------------------
        # compute connectivity
        # ---------------------------------------------------------------------
        # if mi and mi_p have already been computed, reuse it instead
        if len(self._mi) and len(self._mi_p):
            logger.info("    True and permuted mutual-information already "
                        "computed. Use WfConn.clean to reset "
                        "arguments")
            mi, mi_p = self._mi, self._mi_p
        else:
            self._node_prepare_data(dataset)
            mi, mi_p = self._node_compute_mi(
                dataset, self._n_bins, n_perm, n_jobs, random_state)

        # ---------------------------------------------------------------------
        # compute statistics
        # ---------------------------------------------------------------------
        # infer p-values and t-values
        pvalues, tvalues = self._wf_stats.fit(
            mi, mi_p, mcp=mcp, cluster_th=cluster_th, tail=1,
            cluster_alpha=cluster_alpha, inference=self._inference,
            **kw_stats)
        # update internal config
        self.update_cfg(n_perm=n_perm, random_state=random_state,
                        n_bins=n_bins, **self._wf_stats.cfg)

        # ---------------------------------------------------------------------
        # post-processing
        # ---------------------------------------------------------------------
        logger.info(f"    Formatting outputs")
        args = (self._times, dataset.roi_names, self.pairs[0], self.pairs[1],
                'dataarray')
        if isinstance(tvalues, np.ndarray):
            self._tvalues = convert_dfc_outputs(tvalues, *args)
        pvalues = convert_dfc_outputs(pvalues, is_pvalue=True, *args)
        if self._inference is 'rfx':
            mi = np.stack([k.mean(axis=0) for k in mi]).T     # mean mi
        elif self._inference is 'ffx':
            mi = np.concatenate(mi, axis=0).T  # mi
        mi = convert_dfc_outputs(mi, *args)
        # converting outputs
        mi, pvalues = self._attrs_xarray(mi), self._attrs_xarray(pvalues)

        return mi, pvalues

    def clean(self):
        """Clean computations."""
        self._mi, self._mi_p, self._tvalues = [], [], None

    @property
    def mi(self):
        """List of length (n_roi) of true mutual information. Each element of
        this list has a shape of (n_subjects, n_times) if `inference` is 'rfx'
        (1, n_times) if `inference` is 'ffx'."""
        return self._mi

    @property
    def mi_p(self):
        """List of length (n_roi) of permuted mutual information. Each element
        of this list has a shape of (n_perm, n_subjects, n_times) if
        `inference` is 'rfx' (n_perm, 1, n_times) if `inference` is 'ffx'."""
        return self._mi_p

    @property
    def tvalues(self):
        """T-values array of shape (n_times, n_roi) when group level analysis
        is selected."""
        return self._tvalues

    @property
    def wf_stats(self):
        """Get the workflow of statistics."""
        return self._wf_stats
