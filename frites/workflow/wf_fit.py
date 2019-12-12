"""Workflow for Feature Specific Information Transfer."""
import numpy as np

from joblib import Parallel, delayed

from frites import config

from frites.workflow.wf_base import WfBase
from frites.workflow.wf_mi import WfMi
from frites.workflow.wf_stats_ephy import WfStatsEphy

from frites.core import it_fit
from frites.io import (logger, set_log_level, convert_dfc_outputs)
from frites.stats import ttest_1samp


class WfFit(WfBase):
    """Workflow for Feature Specific Information Transfer.

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
    gcrn_per_suj : bool | True
        Apply the Gaussian-rank normalization either per subject (True) or
        across subjects (False).
    mi_method : {'gc', 'bin'}
        Method for computing the mutual information. Use either :

            * 'gc' : gaussian-copula based mutual information. This is the
              fastest method but it can only captures monotonic relationships
              between variables
            * 'bin' : binning-based method that can captures any kind of
              relationships but is much slower and also required to define the
              number of bins to use. Note that if the Numba package is
              installed computations should be much faster

    References
    ----------
    Bim et al. 2019 :cite:`bim2019non`
    """

    def __init__(self, mi_type='cc', inference='rfx', gcrn_per_suj=True,
                 mi_method='gc', verbose=None):  # noqa
        # define the workflow of mi
        self._wf_mi = WfMi(mi_type=mi_type, inference=inference,
                           mi_method=mi_method, gcrn_per_suj=gcrn_per_suj,
                           verbose=False)
        self._mi_type = mi_type
        self._inference = inference
        self._mi_method = mi_method
        set_log_level(verbose)
        self.clean()

        logger.info(
            f"Workflow for computing the FIT ({mi_type} - {mi_method}) and "
            f"statistics ({inference}) has been defined")

    def _node_compute_fit(self, dataset, n_perm, n_jobs, max_delay, directed):
        # ---------------------------------------------------------------------
        # compute mi and permuted mi
        # ---------------------------------------------------------------------
        self._wf_mi.fit(dataset, n_perm=n_perm, n_jobs=n_jobs,
                        output_type='array', stat_method='discard_stats')
        mi = [k.astype(np.float32) for k in self.mi]
        mi_p = [k.astype(np.float32) for k in self.mi_p]

        # ---------------------------------------------------------------------
        # compute fit on true and permuted mi
        # ---------------------------------------------------------------------
        # get the number of pairs (source, target)
        n_roi = len(mi)
        if directed:
            all_s, all_t = np.where(~np.eye(n_roi, dtype=bool))
            tail = 1
        else:
            all_s, all_t = np.triu_indices(n_roi, k=1)
            tail = 0  # two tail test
        pairs = np.c_[all_s, all_t]
        logger.info(f"    Compute FIT (directed={directed}; max_delay="
                    f"{max_delay}; n_pairs={len(all_s)})")
        # get the unique subjects across roi (depends on inference type)
        inference = self._inference
        if inference is 'ffx':
            sujr = [np.array([0])] * n_roi
        else:
            sujr = dataset.suj_roi_u
        times = dataset.times.astype(np.float32)
        # compute FIT on true gcmi values
        cfg_jobs = config.CONFIG["JOBLIB_CFG"]
        arch = Parallel(n_jobs=n_jobs, **cfg_jobs)(delayed(fcn_fit)(
            mi[s], mi[t], mi_p[s], mi_p[t], sujr[s], sujr[t], times,
            max_delay, directed, inference) for s, t in zip(all_s, all_t))
        fit_roi, fitp_roi, fit_m = [list(k) for k in zip(*arch)]
        """
        For sEEG data, the ROI repartition is not the same across subjects.
        Then it's possible that one element of the fit is an empty array. The
        following step clean up empty arrays
        """
        sources, targets, empty = [], [], []
        for k in reversed(range(len(fit_roi))):
            if not fit_roi[k].size:
                fit_roi.pop(k), fitp_roi.pop(k), fit_m.pop(k)  # noqa
                empty += [k]
            else:
                sources += [all_s[k]]
                targets += [all_t[k]]
        sources.reverse(), targets.reverse()  # noqa
        if len(empty):
            logger.info(f"    The FIT inside {len(empty)} have been removed "
                        "because of empty arrays")
        # cache variables
        self._sources, self._targets = np.array(sources), np.array(targets)
        self._tail = tail
        self._fit_roi, self._fitp_roi = fit_roi, fitp_roi
        self._fit_m = fit_m


    def fit(self, dataset, max_delay=0.3, directed=True, n_perm=1000,
            n_jobs=-1, stat_method='rfx_cluster_ttest', mcp='maxstat',
            output_type='3d_dataframe', **kw_stats):
        """Compute the Feature Specific Information transfer and statistics.

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
        max_delay : float | 0.3
            Maximum delay to use for defining the past of the source and target
        directed : bool | True
            Use either a directed FIT (True) or un-directed (False) which is
            defined as FIT(s -> t) - FIT(t -> s)
        n_perm : int | 1000
            Number of permutations to perform in order to estimate the random
            distribution of mi that can be obtained by chance
        n_jobs : int | -1
            Number of jobs to use for parallel computing (use -1 to use all
            jobs)
        stat_method : string | "rfx_cluster_ttest"
            Statistical method to use. For further details, see
            :class:`frites.WfStatsEphy.fit`
        mcp : {'maxstat', 'fdr', 'bonferroni'}
            Method to use for correcting p-values for the multiple comparison
            problem. By default, the maximum-statistics is used.
        output_type : string
            Output format of the returned FIT and p-values. For details, see
            :func:`frites.io.convert_dfc_outputs`. Use either '2d_array',
            '3d_array', '2d_dataframe', '3d_dataframe', 'dataarray'.
        kw_stats : dict | {}
            Additional arguments to pass to the selected statistical method
            selected using the `stat_method` input parameter

        Returns
        -------
        fit, pvalues : array_like
            Array of FIT and p-values. Output types and shapes depends on the
            `output_type` input parameter.

        References
        ----------
        Maris and Oostenveld, 2007 :cite:`maris2007nonparametric`
        """
        # ---------------------------------------------------------------------
        # parameters
        # ---------------------------------------------------------------------
        # if stat_method is None, avoid computing permutations
        inference = self._inference
        if not self._check_stat(inference, stat_method):
            n_perm = 0
        times = dataset.times
        n_times = len(times)
        # rebuild time vector
        max_delay = np.float32(max_delay)
        max_delay_i = n_times - len(np.where(times > times[0] + max_delay)[0])
        self._times = times[max_delay_i:]

        # ---------------------------------------------------------------------
        # compute the FIT
        # ---------------------------------------------------------------------
        # compute fit (only if not already computed)
        if not len(self._fit_roi):
            self._node_compute_fit(dataset, n_perm, n_jobs, max_delay,
                                   directed)
        else:
            logger.warning("    True and permuted FIT already computed. "
                           "Use WfFit.clean() to reset arguments")

        # ---------------------------------------------------------------------
        # statistical evaluation
        # ---------------------------------------------------------------------
        if inference == 'rfx': kw_stats['tail'] = self._tail  # noqa
        self._wf_stats = WfStatsEphy()
        pvalues, tvalues = self._wf_stats.fit(
            self._fit_roi, self._fitp_roi, stat_method=stat_method,
            ttested=True, mcp=mcp, **kw_stats)

        # ---------------------------------------------------------------------
        # post-processing
        # ---------------------------------------------------------------------
        logger.info(f"    Formatting output type ({output_type})")
        args = (self._times, dataset.roi_names, self._sources, self._targets,
                output_type)
        if isinstance(tvalues, np.ndarray):
            self._tvalues = convert_dfc_outputs(tvalues, *args)
        pvalues = convert_dfc_outputs(pvalues, *args)
        if inference is 'ffx':
            fit = np.concatenate(self._fit_roi, axis=0).T  # mi
        elif inference is 'rfx':
            fit = np.stack(self._fit_m, axis=1)     # mean mi
        fit = convert_dfc_outputs(fit, *args)

        return fit, pvalues

    def clean(self):
        """Clean computations."""
        self._sources, self._targets, self._times = None, None, None
        self._fit_roi, self._fitp_roi, self._fit_m = [], [], []
        self._tvalues = None
        self._wf_mi.clean()

    @property
    def sources(self):
        """Get the sources indices of shape (n_pairs,) that have been used."""
        return self._sources

    @property
    def targets(self):
        """Get the targets indices of shape (n_pairs,) that have been used."""
        return self._targets

    @property
    def times(self):
        """Get the time vector."""
        return self._times
    
    @property
    def fit_roi(self):
        """Get the true FIT that have been computed per ROI. This attribute
        is a list of length (n_roi) composed of arrays of shapes
        (1, n_times). For the fixed effect, the 1 is obtained by computing mi
        across subjects then each array of the list is in bits. On the other
        hand, for the random effect the 1 is obtained by computing a one sample
        t-test across subjects. Hence, each array represents t-values per roi.
        """
        return self._fit_roi

    @property
    def fitp_roi(self):
        """Get the permuted FIT that have been computed per ROI. This attribute
        is a list of length (n_roi) composed of arrays of shapes
        (n_perm, 1, n_times). For the fixed effect, the 1 is obtained by
        computing mi across subjects then each array of the list is in bits. On
        the other hand, for the random effect the 1 is obtained by computing a
        one sample t-test across subjects. Hence, each array represents
        t-values per roi."""
        return self._fitp_roi

    @property
    def tvalues(self):
        """T-values array of shape (n_times, n_roi) when group level analysis
        is selected."""
        return self._tvalues
    
    @property
    def mi(self):
        """List of length (n_roi) of true mutual information. Each element of
        this list has a shape of (n_subjects, n_times) if `inference` is 'rfx'
        (1, n_times) if `inference` is 'ffx'."""
        return self._wf_mi._mi

    @property
    def mi_p(self):
        """List of length (n_roi) of permuted mutual information. Each element
        of this list has a shape of (n_perm, n_subjects, n_times) if
        `inference` is 'rfx' (n_perm, 1, n_times) if `inference` is 'ffx'."""
        return self._wf_mi._mi_p




def fcn_fit(x_s, x_t, xp_s, xp_t, suj_s, suj_t, times, max_delay, directed,
            inference):
    """Compute FIT in parallel."""
    # find the unique list of subjects for the source and target
    u_suj = np.unique(np.r_[suj_s, suj_t])
    fit_suj, fitp_suj = [], []
    for s in u_suj:
        # find where the subject is located
        is_source = suj_s == s
        is_target = suj_t == s
        if is_source.any() and is_target.any():
            # select (source, target) data
            x_s_suj = x_s[np.newaxis, is_source, :]
            x_t_suj = x_t[np.newaxis, is_target, :]
            xp_s_suj = xp_s[:, is_source, :]
            xp_t_suj = xp_t[:, is_target, :]
            # FIT on true and permuted gcmi
            _fit_suj = it_fit(x_s_suj, x_t_suj, times, max_delay)[0, ...]
            _fitp_suj = it_fit(xp_s_suj, xp_t_suj, times, max_delay)
            # compute unidirected FIT
            if not directed:
                # compute target -> source
                _fit_ts = it_fit(x_t_suj, x_s_suj, times, max_delay)[0, ...]
                _fitp_ts = it_fit(xp_t_suj, xp_s_suj, times, max_delay)
                # subtract to source -> target
                _fit_suj -= _fit_ts
                _fitp_suj -= _fitp_ts
            # keep the computed (uni/bi)directed FIT
            fit_suj += [_fit_suj]
            fitp_suj += [_fitp_suj]
    # if not subjects, return empty arrays
    if not len(fit_suj):
        return np.array([]), np.array([]), np.array([])
    """
    Returns depends on inference types :
    * Fixed-effect : do nothing because it has already been computed across
      subjects, just returns [0] because there's only a "single-subject"
    * Random-effect : compute the t-test against the mean of permutations in
      order to decrease the size in memory.
    """
    if inference == 'ffx':
        return fit_suj[0], fitp_suj[0], np.array([])
    elif inference == 'rfx':
        # concatenate across remaining axis
        fit_suj_cat = np.concatenate(fit_suj, axis=0)
        fitp_suj_cat = np.concatenate(fitp_suj, axis=1)
        fit_m = fit_suj_cat.mean(axis=0)
        # skip t-test if there's only a single subject (sEEG)
        if fit_suj_cat.shape[0] == 1:
            return np.array([]), np.array([]), np.array([])
        # compute t-test across subject dimension
        pop_mean_surr = np.mean([k.mean() for k in fitp_suj_cat])  # sEEG patch
        tt = ttest_1samp(fit_suj_cat, pop_mean_surr, axis=0)
        ttp = ttest_1samp(fitp_suj_cat, pop_mean_surr, axis=1)

        return tt[np.newaxis, :], ttp[:, np.newaxis, :], fit_m
