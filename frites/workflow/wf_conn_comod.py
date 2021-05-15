"""Workflow of connectivity."""
import numpy as np
import xarray as xr

from mne.utils import ProgressBar

from frites.io import (set_log_level, logger)
from frites.core import permute_mi_trials
from frites.utils import parallel_func, kernel_smoothing
from frites.workflow.wf_stats import WfStats
from frites.workflow.wf_base import WfBase
from frites.estimator import GCMIEstimator


class WfConnComod(WfBase):
    """Workflow of instantaneous pairwise comodulations and statistics.

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
    estimator : MIEstimator | None
        Estimator of mutual-information. If None, the Gaussian-Copula is used
        instead. Note that here, since the mutual information is computed
        between two time-series coming from two brain regions, the estimator
        should has a mi_type='cc'
    kernel : array_like | None
        Kernel for smoothing true and permuted MI. For example, use
        np.hanning(3) for a 3 time points smoothing or np.ones((3)) for a
        moving average

    References
    ----------
    Friston et al., 1996, 1999 :cite:`friston1996detecting,friston1999many`

    See also
    --------
    conn_get_pairs
    conn_reshape_undirected
    """

    def __init__(self, inference='rfx', estimator=None, kernel=None,
                 verbose=None):
        """Init."""
        WfBase.__init__(self)
        assert inference in ['ffx', 'rfx'], (
            "'inference' input parameter should either be 'ffx' or 'rfx'")
        self._mi_type = 'cc'
        if estimator is None:
            estimator = GCMIEstimator(mi_type='cc', copnorm=False,
                                      verbose=verbose)
        assert estimator.settings['mi_type'] == self._mi_type
        self._copnorm = isinstance(estimator, GCMIEstimator)
        self._inference = inference
        self.estimator = estimator
        self._gcrn = inference == 'rfx'
        self._kernel = kernel
        set_log_level(verbose)
        self.clean()
        self._wf_stats = WfStats(verbose=verbose)
        # update internal config
        self.attrs.update(dict(mi_type=self._mi_type, inference=inference,
                               kernel=kernel))

        logger.info(f"Workflow for computing comodulations between distant "
                    f"brain areas ({inference})")

    def _node_compute_mi(self, dataset, n_perm, n_jobs, random_state):
        """Compute mi and permuted mi.

        Permutations are performed by randomizing the target roi. For the fixed
        effect, this randomization is performed across subjects. For the random
        effect, the randomization is performed per subject.
        """
        # get the function for computing mi
        core_fun = self.estimator.get_function()
        # get the pairs for computing mi
        df_conn, _ = dataset.get_connectivity_pairs(
            directed=False, as_blocks=True)
        sources, targets = df_conn['sources'], df_conn['targets']
        self._pair_names = np.concatenate(df_conn['names'])
        n_pairs = len(self._pair_names)
        # parallel function for computing permutations
        parallel, p_fun = parallel_func(comod, n_jobs=n_jobs, verbose=False)
        pbar = ProgressBar(range(n_pairs), mesg='Estimating MI')
        # evaluate true mi
        mi, mi_p, inf = [], [], self._inference
        kw_get = dict(mi_type=self._mi_type, copnorm=self._copnorm,
                      gcrn_per_suj=self._gcrn)
        for n_s, s in enumerate(sources):
            # get source data
            da_s = dataset.get_roi_data(s, **kw_get)
            suj_s = da_s['subject'].data
            for t in targets[n_s]:
                # get target data
                da_t = dataset.get_roi_data(t, **kw_get)
                suj_t = da_t['subject'].data

                # compute mi
                _mi = comod(da_s.data, da_t.data, suj_s, suj_t, inf,
                            core_fun)
                # get the randomize version of y
                y_p = permute_mi_trials(suj_t, inference=inf, n_perm=n_perm)
                # run permutations using the randomize regressor
                _mi_p = parallel(p_fun(
                    da_s.data, da_t.data[..., y_p[p]], suj_s, suj_t, inf,
                    core_fun) for p in range(n_perm))
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

    def fit(self, dataset, mcp='cluster', n_perm=1000, cluster_th=None,
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
            DataArray of mean mutual information and p-values both of shapes
            (n_times, n_pairs)

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
        # get important dataset's variables
        self._times = dataset.times

        # ---------------------------------------------------------------------
        # compute connectivity
        # ---------------------------------------------------------------------
        # if mi and mi_p have already been computed, reuse it instead
        if len(self._mi) and len(self._mi_p):
            logger.info("    True and permuted mutual-information already "
                        "computed. Use WfConnComod.clean to reset "
                        "arguments")
            mi, mi_p = self._mi, self._mi_p
        else:
            mi, mi_p = self._node_compute_mi(
                dataset, n_perm, n_jobs, random_state)

        # ---------------------------------------------------------------------
        # compute statistics
        # ---------------------------------------------------------------------
        # infer p-values and t-values
        pvalues, tvalues = self._wf_stats.fit(
            mi, mi_p, mcp=mcp, cluster_th=cluster_th, tail=1,
            cluster_alpha=cluster_alpha, inference=self._inference,
            **kw_stats)
        # update internal config
        self.attrs.update(dict(n_perm=n_perm, random_state=random_state))
        self.attrs.update(self._wf_stats.attrs)

        # ---------------------------------------------------------------------
        # post-processing
        # ---------------------------------------------------------------------
        logger.info("    Formatting outputs")
        if isinstance(tvalues, np.ndarray):
            self._tvalues = self._xr_conversion(tvalues, 'tvalues')
        pvalues = self._xr_conversion(pvalues, 'pvalues')
        if self._inference == 'rfx':
            mi = np.stack([k.mean(axis=0) for k in mi]).T
        elif self._inference == 'ffx':
            mi = np.concatenate(mi, axis=0).T
        mi = self._xr_conversion(mi, 'mi')

        return mi, pvalues

    def clean(self):
        """Clean computations."""
        self._mi, self._mi_p, self._tvalues = [], [], None

    def _xr_conversion(self, x, name):
        """Xarray conversion."""
        # build dimension order
        dims = ['times', 'roi']
        coords = [self._times, self._pair_names]
        # build xarray
        da = xr.DataArray(x, dims=dims, coords=coords)
        # wrap with workflow's attributes
        da = self.attrs.wrap_xr(da, name=name)
        return da

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


def comod(x_1, x_2, suj_1, suj_2, inference, fun):
    """I(C; C) for rfx.

    The returned mi array has a shape of (n_subjects, n_times) if inference is
    "rfx", (1, n_times) if "ffx".
    """
    # proper shape of the regressor
    n_times, _, n_trials = x_1.shape
    # compute mi across (ffx) or per subject (rfx)
    if inference == 'ffx':
        mi = fun(x_1, x_2)
    elif inference == 'rfx':
        # get subject informations
        suj_u = np.intersect1d(suj_1, suj_2)
        n_subjects = len(suj_u)
        # compute mi per subject
        mi = np.zeros((n_subjects, n_times), dtype=float)
        for n_s, s in enumerate(suj_u):
            is_suj_1 = suj_1 == s
            is_suj_2 = suj_2 == s
            mi[n_s, :] = fun(x_1[..., is_suj_1], x_2[..., is_suj_2])

    return mi
