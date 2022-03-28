"""Workflow for computing MI and evaluate statistics."""
from copy import deepcopy

import numpy as np
import xarray as xr

from mne.utils import ProgressBar

from frites.io import set_log_level, logger
from frites.workflow.wf_stats import WfStats
from frites.workflow.wf_base import WfBase
from frites.estimator import GCMIEstimator, ResamplingEstimator
from frites.utils import parallel_func, kernel_smoothing
from frites.stats import dist_to_ci, permute_mi_vector, bootstrap_partitions


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
    estimator : MIEstimator | None
        Estimator of mutual-information. If None, the Gaussian-Copula is used
        instead.
    kernel : array_like | None
        Kernel for smoothing true and permuted MI. For example, use
        np.hanning(3) for a 3 time points smoothing or np.ones((3)) for a
        moving average

    References
    ----------
    Friston et al., 1996, 1999 :cite:`friston1996detecting,friston1999many`
    """

    def __init__(self, mi_type='cc', inference='rfx', estimator=None,
                 kernel=None, verbose=None):
        """Init."""
        WfBase.__init__(self)
        assert mi_type in ['cc', 'cd', 'ccd'], (
            "'mi_type' input parameter should either be 'cc', 'cd', 'ccd'")
        assert inference in ['ffx', 'rfx'], (
            "'inference' input parameter should either be 'ffx' or 'rfx'")
        self._mi_type = mi_type
        self._inference = inference
        if estimator is None:
            estimator = GCMIEstimator(mi_type=mi_type, copnorm=False,
                                      verbose=verbose)
        elif isinstance(estimator, ResamplingEstimator):
            raise NotImplementedError("Resampling estimators are not supported"
                                      " in WfMi")
        assert estimator.settings['mi_type'] == self._mi_type
        self.estimator = estimator
        self._copnorm = isinstance(estimator, GCMIEstimator)
        self._gcrn = inference == 'rfx'
        self._kernel = kernel
        set_log_level(verbose)
        self.clean()
        self._wf_stats = WfStats(verbose=verbose)
        # update internal config
        self.attrs.update(dict(
            mi_type=mi_type, inference=inference, kernel=kernel))

        logger.info(
            f"Workflow for computing mutual information (inference={inference}"
            f", mi_type={mi_type}, copnorm={self._copnorm})")

    def _node_compute_mi(self, dataset, n_perm, n_jobs, random_state):
        """Compute mi and permuted mi.

        Permutations are performed by randomizing the regressor variable. For
        the fixed effect, this randomization is performed across subjects. For
        the random effect, the randomization is performed per subject.
        """
        # get the function for computing mi
        mi_fun = self.estimator.get_function()
        # get x, y, z and subject names per roi
        if dataset._mi_type != self._mi_type:
            assert TypeError(f"Your dataset doesn't allow to compute the mi "
                             f"{self._mi_type}. Allowed mi is "
                             f"{dataset._mi_type}")
        # get data variables
        n_roi = len(self._roi)
        # evaluate true mi
        logger.info(f"    Evaluate true and permuted mi (n_perm={n_perm}, "
                    f"n_jobs={n_jobs})")
        # parallel function for computing permutations
        parallel, p_fun = parallel_func(mi_fun, n_jobs=n_jobs, verbose=False)
        pbar = ProgressBar(range(n_roi), mesg='Estimating MI')
        # evaluate permuted mi
        mi, mi_p = [], []
        for r in range(n_roi):
            # get the data of selected roi
            da = dataset.get_roi_data(
                self._roi[r], copnorm=self._copnorm, mi_type=self._mi_type,
                gcrn_per_suj=self._gcrn)
            x, y, suj = da.data, da['y'].data, da['subject'].data
            kw_mi = dict()
            # cmi and categorical MI
            if 'z' in list(da.coords):
                kw_mi['z'] = da['z'].data
            if self._inference == 'rfx':
                kw_mi['categories'] = suj

            # compute the true mi
            _mi = mi_fun(x, y, **kw_mi)
            # get the randomize version of y
            y_p = permute_mi_vector(
                y, suj, mi_type=self._mi_type, inference=self._inference,
                n_perm=n_perm, random_state=random_state)
            # run permutations using the randomize regressor
            _mi_p = parallel(p_fun(x, y_p[p], **kw_mi) for p in range(n_perm))
            _mi_p = np.asarray(_mi_p)

            # kernel smoothing
            if isinstance(self._kernel, np.ndarray):
                _mi = kernel_smoothing(_mi, self._kernel, axis=-1)
                _mi_p = kernel_smoothing(_mi_p, self._kernel, axis=-1)

            mi += [_mi]
            mi_p += [_mi_p]
            pbar.update_with_increment_value(1)

        self._mi, self._mi_p = mi, mi_p

        return mi, mi_p

    def fit(self, dataset=None, mcp='cluster', n_perm=1000, cluster_th=None,
            cluster_alpha=0.05, n_jobs=-1, random_state=None, **kw_stats):
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
            A dataset instance. If the workflow has already been fitted, then
            this parameter can remains to None.
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
        n_jobs : int | -1
            Number of jobs to use for parallel computing (use -1 to use all
            jobs)
        random_state : int | None
            Fix the random state of the machine (use it for reproducibility).
            If None, a random state is randomly assigned.
        kw_stats : dict | {}
            Additional arguments are sent to
            :py:class:`frites.workflow.WfStats.fit`

        Returns
        -------
        mi, pvalues : array_like
            DataArray of mutual information and p-values both of shapes
            (n_times, n_roi). If `inference` is 'ffx' the mi represents the MI
            computed across subjects while if it is 'rfx' it's the mean across
            subjects.

        References
        ----------
        Maris and Oostenveld, 2007 :cite:`maris2007nonparametric`
        """
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
            # don't compute permutations if mcp is either nostat / None
            if mcp in ['noperm', None]:
                n_perm = 0

            # get needed dataset's informations
            self._times, self._roi = dataset.times, dataset.roi_names
            self._mi_dims = dataset._mi_dims
            self._mi_coords = dict()
            for k in self._mi_dims:
                if k != 'roi':
                    self._mi_coords[k] = dataset.x[0].coords[k].data
                else:
                    self._mi_coords['roi'] = self._roi
            self._df_rs, self._n_subjects = dataset.df_rs, dataset._n_subjects

            # compute mi and permutations
            mi, mi_p = self._node_compute_mi(
                dataset, n_perm, n_jobs, random_state)
        """
        For information transfer (e.g FIT) we only need to compute the true and
        permuted mi but then, the statistics at the local representation mcp
        are discarded in favor of statistics on the information transfer
        """
        if mcp == 'nostat':
            logger.debug("Permutations computed. Stop there")
            return None

        # ---------------------------------------------------------------------
        # compute statistics
        # ---------------------------------------------------------------------
        # get additional stat arguments
        kw_stats['tail'] = kw_stats.get('tail', 1)
        # infer p-values and t-values
        pvalues, tvalues = self._wf_stats.fit(
            mi, mi_p, cluster_th=cluster_th, inference=self._inference,
            mcp=mcp, cluster_alpha=cluster_alpha, **kw_stats)
        # update attributes
        self.attrs.update(self._wf_stats.attrs)
        self.attrs.update(dict(n_perm=n_perm, random_state=random_state))

        # ---------------------------------------------------------------------
        # postprocessing and conversions
        # ---------------------------------------------------------------------
        # tvalues conversion
        if isinstance(tvalues, np.ndarray):
            self._tvalues = self._xr_conversion(tvalues, 'tvalues')
        # mean mi across subjects
        if self._inference == 'rfx':
            logger.info("    Mean mi across subjects")
            mi = [k.mean(axis=0, keepdims=True) for k in mi]
        mi = np.moveaxis(np.concatenate(mi, axis=0), 0, -1)
        # dataarray conversion
        mi = self._xr_conversion(mi, 'mi')
        pv = self._xr_conversion(pvalues, 'pvalues')

        return mi, pv

    def _xr_conversion(self, x, name):
        """Xarray conversion."""
        # build dimension order
        dims = ['times', 'roi']
        supp_dim = [k for k in self._mi_dims if k not in dims]
        dims = supp_dim + dims
        # build coordinates
        coords = [self._mi_coords[k] for k in dims]
        # build xarray
        da = xr.DataArray(x, dims=dims, coords=coords)
        # wrap with workflow's attributes
        da = self.attrs.wrap_xr(da, name=name)
        return da

    def conjunction_analysis(self, p=.05, mcp='cluster', cluster_th=None,
                             cluster_alpha=0.05):
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
        pv_s = {}
        for s in range(self._n_subjects):
            # reconstruct the mi and mi_p of each subject
            mi_s, mi_ps, roi_s = [], [], []
            for n_r, r in enumerate(self._roi):
                suj_roi_u = np.array(self._df_rs.loc[r, 'subjects'])
                if s not in suj_roi_u: continue  # noqa
                is_suj = suj_roi_u == s
                mi_s += [self._mi[n_r][is_suj, :]]
                mi_ps += [self._mi_p[n_r][:, is_suj, :]]
                roi_s += [self._roi[n_r]]

            # perform the statistics
            _pv_s = self._wf_stats.fit(
                mi_s, mi_ps, mcp=mcp, cluster_th=cluster_th, tail=1,
                cluster_alpha=cluster_alpha, inference='ffx')[0]
            # dataarray conversion
            pv_s[s] = xr.DataArray(_pv_s < p, dims=('times', 'roi'),
                                   coords=(self._times, roi_s))
        # cross-subjects conjunction
        conj_ss = xr.Dataset(pv_s).to_array('subject')
        conj_ss.name = 'Single subject conjunction'
        conj = conj_ss.sum('subject')
        conj.name = 'Across subjects conjunction'
        # add attributes to the dataarray
        attrs = dict(p=p, cluster_th=cluster_th, cluster_alpha=cluster_alpha,
                     mcp=mcp)
        for k, v in attrs.items():
            v = 'none' if v is None else v
            conj[k], conj_ss[k] = v, v

        return conj_ss, conj

    def confidence_interval(self, dataset, ci=95, n_boots=200, rfx_es='mi',
                            n_jobs=-1, random_state=None, verbose=None):
        """Estimate the empirical confidence interval.

        Parameters
        ----------
        dataset : :class:`frites.dataset.DatasetEphy`
            A dataset instance. If the workflow has already been fitted, then
            this parameter can remains to None.
        ci : float, list | 95
            Confidence level to use in percentage. Use either a single float
            (e.g. 95, 99 etc.) or a list of floats (e.g. [95, 99])
        n_boots : int | 200
            Number of resampling to perform
        rfx_es : {'mi', 'tvalues'}
            For the RFX model, specify whether the confidence interval has to
            be estimated on a measure of effect size in bits ('mi') or on
            second-level t-test ('tvalues')
        n_jobs : int | -1
            Number of jobs to use for parallel computing (use -1 to use all
            jobs)
        random_state : int | None
            Fix the random state of the machine (use it for reproducibility).
            If None, a random state is randomly assigned.

        Returns
        -------
        ci : xr.DataArray
            Confidence interval array of shape (n_ci, 2, n_times, n_roi) where
            n_ci describe the number of confidence levels define with the
            input parameter `ci`, and 2 represents the lower and upper bounds
            of the confidence interval
        """
        set_log_level(verbose)

        # check inputs
        if isinstance(ci, (int, float)):
            ci = [ci]
        assert isinstance(ci, (list, tuple, np.ndarray))
        assert isinstance(n_boots, int)
        for k in ci:
            assert isinstance(k, (int, float)) and (0 < k < 100)
        assert len(self._mi) and len(self._mi_p), (
            "You've to lauched the workflow (`fit()`) before being able to "
            "perform the conjunction analysis.")
        assert rfx_es in ['mi', 'tvalues']

        logger.info("Estimation of the empirical confidence interval "
                    f"(levels={ci}; n_boots={n_boots}")

        """
        For the RFX, the CI on t-values is estimated at the second level i.e.
        by bootstraping the subjects. I'm not sure why, but CI at the trial
        level (i.e. resampling trials per subjects) leads to t-values bellow
        the real effect size.
        """
        if (self._inference == 'rfx') and (rfx_es == 'tvalues'):
            from frites.stats import ttest_1samp

            logger.info("    Resampling at the second level t-test")

            # get whether the same partition should be used across roi or not
            n_suj_roi = np.array([k.shape[0] for k in self.mi])
            part_fix = np.all(n_suj_roi == n_suj_roi[0])

            # create the resampling partitions for each roi (seeg)
            if part_fix:  # fix partitions
                partitions = bootstrap_partitions(
                    max(n_suj_roi), n_partitions=n_boots,
                    random_state=random_state)
            else:         # flexible partitions
                partitions = []
                for k in range(len(self.mi)):
                    _part = bootstrap_partitions(
                        self.mi[k].shape[0], n_partitions=n_boots,
                        random_state=random_state)
                    partitions.append(_part)

            # create the progress bar
            pbar = ProgressBar(range(n_boots), mesg='Estimating CI')

            # get t-test related variables
            s_hat = self._wf_stats.attrs['ttest_sigma']

            tt = []
            for n_p in range(n_boots):
                # resample the mi
                mi_rsh, mi_p_rsh, n_elements = [], 0., 0.
                for n_r, (mi, mi_p) in enumerate(zip(self.mi, self.mi_p)):
                    # get the partition
                    p = partitions[n_p] if part_fix else partitions[n_r][n_p]
                    # reshape mi and mi_p
                    mi_rsh.append(mi[p, :])
                    _mi_p_rsh = mi_p[:, p, :]
                    mi_p_rsh += _mi_p_rsh.sum()
                    n_elements += np.prod(_mi_p_rsh.shape)

                # computes pop_mean on resampled t-values
                pop_mean = mi_p_rsh / n_elements

                # compute the t-test on this partition
                mi_var = max([np.var(k, axis=0, ddof=1).max() for k in mi_rsh])
                sigma = s_hat * mi_var
                _tt = np.stack([ttest_1samp(
                    k, pop_mean, axis=0, implementation='mne',
                    method='absolute', sigma=sigma) for k in mi_rsh], axis=0)
                tt.append(_tt)

                pbar.update_with_increment_value(1)

            # compute ci
            tt = np.stack(tt, axis=0)
            ci_all = [dist_to_ci(tt[:, [r], :], cis=ci) for r in range(
                tt.shape[1])]
            ci_all = np.stack(ci_all, axis=-1)

            # xarray formatting
            x_ci = xr.DataArray(
                ci_all, dims=('ci', 'bound', 'times', 'roi'),
                coords=(ci, ['low', 'high'], self._mi_coords['times'],
                        self._mi_coords['roi']))
            return x_ci

        logger.info("    Resampling at the first level")

        """
        For the FFX / RFX + rfx_es = 'mi', the resampling is performed at the
        single trial level.
        """

        # get the function for computing mi
        mi_fun = self.estimator.get_function()
        n_roi = len(self._roi)
        if self._inference == 'rfx':
            pop_mean = self.attrs['ttest_pop_mean']
        else:
            pop_mean = None

        # get x, y, z and subject names per roi
        if dataset._mi_type != self._mi_type:
            assert TypeError(f"Your dataset doesn't allow to compute the mi "
                             f"{self._mi_type}. Allowed mi is "
                             f"{dataset._mi_type}")

        # evaluate true mi
        logger.info(f"    Evaluate the confidence interval (ci={ci}%, "
                    f"n_boots={n_boots}, n_jobs={n_jobs})")

        # define the bootstraping function
        def boot_fcn(x, y, part, **kw_mi):
            # resample the subject labels and z variable (cmi)
            kw = {k: v[part] for k, v in kw_mi.items()}

            return mi_fun(x[..., part], y[part], **kw)

        # parallel function for computing bootstrap
        parallel, p_fun = parallel_func(boot_fcn, n_jobs=n_jobs, verbose=False)
        pbar = ProgressBar(range(n_roi), mesg='Estimating CI')

        # evaluate permuted mi
        x_ci = []
        for r in range(n_roi):
            # get the data of selected roi
            da = dataset.get_roi_data(
                self._roi[r], copnorm=self._copnorm, mi_type=self._mi_type,
                gcrn_per_suj=self._gcrn)
            x, y, suj = da.data, da['y'].data, da['subject'].data
            kw_mi = dict()

            # cmi and categorical MI
            if 'z' in list(da.coords):
                kw_mi['z'] = da['z'].data
            if self._inference == 'rfx':
                kw_mi['categories'] = suj

            # build the group variable
            groups = tuple(v for v in kw_mi.values())

            # define the bootstrap partitions
            partitions = bootstrap_partitions(
                len(y), *groups, n_partitions=n_boots,
                random_state=random_state)

            # compute the true mi
            _mi_b = parallel(p_fun(
                x, y, p, **kw_mi) for p in partitions)

            # compute ci
            _x_ci = dist_to_ci(
                np.asarray(_mi_b), cis=ci, inference=self._inference,
                pop_mean=pop_mean, rfx_es='mi'
            )
            x_ci.append(_x_ci)
            pbar.update_with_increment_value(1)
        x_ci = np.stack(x_ci, axis=-1)

        # xarray conversion
        x_ci = xr.DataArray(
            x_ci, dims=('ci', 'bound', 'times', 'roi'),
            coords=(ci, ['low', 'high'], self._mi_coords['times'],
                    self._mi_coords['roi']))

        return x_ci

    def get_params(self, *params):
        """Get formatted parameters.

        This method can be used to get internal arrays formatted as xarray
        DataArray.

        Parameters
        ----------
        params : string
            Internal array names to get as xarray DataArray. You can use :

                * 'tvalues' : DataArray of t-values of shape (n_times, n_roi).
                  Only possible with RFX inferences
                * 'mi_ss' : DataArray of single subject mutual-information of
                  shape (n_subjects, n_times, n_roi)
                * 'perm_ss' : DataArray of computed permutations of shape
                  (n_perm, n_subjects, n_times, n_roi)
                * 'perm_' : DataArray of maximum computed permutations of
                  shape (n_perm,)
        """
        # get coordinates
        times, roi, df_rs = self._times, self._roi, self._df_rs
        if self._inference == 'ffx':
            suj = [np.array([-1])] * len(roi)
        elif self._inference == 'rfx':
            suj = [np.array(df_rs.loc[r, 'subjects']) for r in roi]
        n_perm = self._mi_p[0].shape[0]
        perm = np.arange(n_perm)
        # loop over possible outputs
        outs = []
        for param in params:
            assert isinstance(param, str)
            logger.info(f'    Formatting array {param}')
            if param == 'tvalues':
                da = self._tvalues
            elif param == 'mi_ss':
                mi = dict()
                for n_r, r in enumerate(roi):
                    mi[r] = xr.DataArray(
                        self._mi[n_r], coords=(suj[n_r], times),
                        dims=('subject', 'times'))
                da = xr.Dataset(mi).to_array('roi')
                da = da.transpose('subject', 'times', 'roi')
            elif param == 'perm_ss':
                mi = dict()
                for n_r, r in enumerate(roi):
                    mi[r] = xr.DataArray(
                        self._mi_p[n_r], dims=('perm', 'subject', 'times'),
                        coords=(perm, suj[n_r], times))
                da = xr.Dataset(mi).to_array('roi')
                da = da.transpose('perm', 'subject', 'times', 'roi')
            elif param == 'perm_':
                mi_p = np.r_[tuple([k.ravel() for k in self._mi_p])]
                mi_p.sort()
                da = xr.DataArray(mi_p[-n_perm:], dims=('perm',),
                                  coords=(perm,))
            else:
                raise ValueError(f"Parameter {param} not found")
            # add workflow attributes
            self.attrs.wrap_xr(da, name=param)
            outs += [da]

        # fix returning single output
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def clean(self):
        """Clean computations."""
        self._mi, self._mi_p, self._tvalues = [], [], None

    def copy(self):
        """Return copy of WfMi instance.

        Returns
        -------
        epochs : instance of WfMi
            A copy of the object.
        """
        return deepcopy(self)

    def __deepcopy__(self, memodict):
        """Make a deepcopy."""
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            # drop_log is immutable and _raw is private (and problematic to
            # deepcopy)
            if k in ('drop_log', '_raw', '_times_readonly'):
                memodict[id(v)] = v
            else:
                v = deepcopy(v, memodict)
            result.__dict__[k] = v
        return result

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
