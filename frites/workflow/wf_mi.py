"""Workflow for computing MI and evaluate statistics."""
import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from frites import config
from frites.io import set_log_level, logger
from frites.core import get_core_mi_fun, permute_mi_vector
from frites.workflow.wf_stats_ephy import WfStatsEphy
from frites.workflow.wf_base import WfBase


class WfMi(WfBase):
    """Workflow of local mutual-information and statistics.

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

    def __init__(self, mi_type='cc', inference='rfx', mi_method='gc',
                 kernel=None, verbose=None):
        """Init."""
        WfBase.__init__(self)
        assert mi_type in ['cc', 'cd', 'ccd'], (
            "'mi_type' input parameter should either be 'cc', 'cd', 'ccd'")
        assert inference in ['ffx', 'rfx'], (
            "'inference' input parameter should either be 'ffx' or 'rfx'")
        assert mi_method in ['gc', 'bin'], (
            "'mi_method' input parameter should either be 'gc' or 'bin'")
        self._mi_type = mi_type
        self._inference = inference
        self._mi_method = mi_method
        self._need_copnorm = mi_method == 'gc'
        self._gcrn = inference is 'rfx'
        self._kernel = kernel
        set_log_level(verbose)
        self.clean()
        self._wf_stats = WfStatsEphy(verbose=verbose)
        # update internal config
        self.update_cfg(mi_type=mi_type, inference=inference,
            mi_method=mi_method, kernel=kernel)

        logger.info(f"Workflow for computing mutual information ({inference} -"
                    f" {mi_method} - {mi_type})")

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

        Permutations are performed by randomizing the regressor variable. For
        the fixed effect, this randomization is performed across subjects. For
        the random effect, the randomization is performed per subject.
        """
        # get the function for computing mi
        mi_fun = get_core_mi_fun(self._mi_method)[self._mi_type]
        assert f"mi_{self._mi_method}_ephy_{self._mi_type}" == mi_fun.__name__
        # get x, y, z and subject names per roi
        if dataset._mi_type != self._mi_type:
            assert TypeError(f"Your dataset doesn't allow to compute the mi "
                             f"{self._mi_type}. Allowed mi is "
                             f"{dataset._mi_type}")
        x, y, z, suj = dataset.x, dataset.y, dataset.z, dataset.suj_roi
        n_roi, inf = dataset.n_roi, self._inference
        # evaluate true mi
        logger.info(f"    Evaluate true and permuted mi (n_perm={n_perm}, "
                    f"n_jobs={n_jobs})")
        mi = [mi_fun(x[k], y[k], z[k], suj[k], inf,
                     n_bins=n_bins) for k in range(n_roi)]
        # get joblib configuration
        cfg_jobs = config.CONFIG["JOBLIB_CFG"]
        # evaluate permuted mi
        mi_p = []
        for r in range(n_roi):
            # get the randomize version of y
            y_p = permute_mi_vector(y[r], suj[r], mi_type=self._mi_type,
                                    inference=self._inference, n_perm=n_perm)
            # run permutations using the randomize regressor
            _mi = Parallel(n_jobs=n_jobs, **cfg_jobs)(delayed(mi_fun)(
                x[r], y_p[p], z[r], suj[r], inf,
                n_bins=n_bins) for p in range(n_perm))
            mi_p += [np.asarray(_mi)]
        # smoothing
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
                * 'nostat' : permutations are computed but no statistics are
                  performed
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
            Additional arguments are sent to
            :py:class:`frites.dataset.WfStatsEphy.fit`

        Returns
        -------
        mi, pvalues : array_like
            DataArray of mutual information and p-values. If `inference` is
            'ffx' the mi represents the MI computed across subjects while if it
            is 'rfx' it's the mean across subjects.

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
        # compute mutual information
        # ---------------------------------------------------------------------
        # if mi and mi_p have already been computed, reuse it instead
        if len(self._mi) and len(self._mi_p):
            logger.info("    True and permuted mutual-information already "
                        "computed. Use WfMi.clean to reset "
                        "arguments")
            mi, mi_p = self._mi, self._mi_p
        else:
            self._node_prepare_data(dataset)
            mi, mi_p = self._node_compute_mi(
                dataset, self._n_bins, n_perm, n_jobs, random_state)
        """
        For information transfer (e.g FIT) we only need to compute the true and
        permuted mi but then, the statistics at the local representation mcp
        are discarded in favor of statistics on the information transfer
        """
        if mcp is 'nostat':
            logger.debug("Permutations computed. Stop there")
            return None

        # ---------------------------------------------------------------------
        # 4d reshaping before the stats
        # ---------------------------------------------------------------------
        self._reshape = dataset._reshape
        if self._reshape is not None:
            logger.debug(f"    reshaping before computing statistics")
            n_f, n_t = self._reshape
            for k in range(len(mi)):
                n_p, n_s, _ = mi_p[k].shape
                mi[k] = mi[k].reshape(n_s, n_f, n_t)
                mi_p[k] = mi_p[k].reshape(n_p, n_s, n_f, n_t)

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
        # postprocessing and conversions
        # ---------------------------------------------------------------------
        # tvalues conversion
        if isinstance(tvalues, np.ndarray):
            self._tvalues = self._xr_conversion(tvalues)
        # mean mi across subjects
        if self._inference is 'rfx':
            logger.info("    Mean mi across subjects")
            mi = [k.mean(axis=0, keepdims=True) for k in mi]
        mi = np.moveaxis(np.concatenate(mi, axis=0), 0, -1)
        # dataarray conversion
        mi = self._xr_conversion(mi)
        pv = self._xr_conversion(pvalues)

        return mi, pv

    def _xr_conversion(self, x):
        """Xarray conversion."""
        times, roi = self._times, self._roi
        if x.ndim == 2:
            x_da = xr.DataArray(x, dims=('times', 'roi'), coords=(times, roi))
        elif x.ndim == 3:
            freqs = np.arange(x.shape[0])
            x_da = xr.DataArray(x, dims=('freqs', 'times', 'roi'),
                                coords=(freqs, times, roi))
        self._attrs_xarray(x_da)

        return x_da


    def conjunction_analysis(self, dataset, p=.05, mcp='cluster',
                             cluster_th=None, cluster_alpha=0.05):
        """Perform a conjunction analysis.

        This method can be used in order to determine the number of subjects
        that present a significant effect at a given significiency threshold.
        Note that in order to work, the workflow of mutual information must
        have already been launched using the
        :py:class:`frites.workflow.WfMi.fit`.


        .. warning::

            In order to work this method require that the workflow has been
            defined with `inference='rfx'` so that MI are computed per subject

        Parameters
        ----------
        dataset : :class:`frites.dataset.DatasetEphy`
            A dataset instance. Note that it should be the same dataset used
            with the fit() method.
        p : float | 0.05
            Significiency threshold to find significant effect per subject.
        kwargs : dict | {}
            Optional arguments are the same as
            :py:class:`frites.workflow.WfMi.fit` method.

        Returns
        -------
        conj_ss : array_like
            DataArray of shape (n_subjects, n_times, n_roi) describing where
            each subject have significant MI
        conj : array_like
            DataArray of shape (n_times, n_roi) describing the number of
            subjects that have a significant MI
        """
        # input checking
        assert self._inference == 'rfx', (
            "Conjunction analysis are only possible when the MI has been "
            "computed per subject (inference='rfx')")
        assert len(self._mi) and len(self._mi_p), (
            "You've to lauched the workflow (`fit()`) before being able to "
            "perform the conjunction analysis.")

        # retrieve the original number of subjects
        n_roi = len(self._mi)
        pv_s = {}
        for s in range(dataset.n_subjects):
            # reconstruct the mi and mi_p of each subject
            mi_s, mi_ps, roi_s = [], [], []
            for r in range(n_roi):
                suj_roi_u = dataset.suj_roi_u[r]
                if s not in suj_roi_u: continue  # noqa
                is_suj = suj_roi_u == s
                mi_s += [self._mi[r][is_suj, :]]
                mi_ps += [self._mi_p[r][:, is_suj, :]]
                roi_s += [self._roi[r]]

            # perform the statistics
            _pv_s = self._wf_stats.fit(
            mi_s, mi_ps, mcp=mcp, cluster_th=cluster_th, tail=1,
            cluster_alpha=cluster_alpha, inference='ffx')[0]
            # dataarray conversion
            pv_s[s] = xr.DataArray(_pv_s < p, dims=('times', 'roi'),
                                   coords=(self._times, roi_s))
        # cross-subjects conjunction
        conj_ss = xr.Dataset(pv_s).to_array('subject')
        conj = conj_ss.sum('subject')
        # add attributes to the dataarray
        attrs = dict(p=p, cluster_th=cluster_th, cluster_alpha=cluster_alpha,
                     mcp=mcp)
        for k, v in attrs.items():
            v = 'none' if v is None else v
            conj[k], conj_ss[k] = v, v

        return conj_ss, conj

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
