"""Workflow for computing MI and evaluate statistics."""
import logging

import numpy as np
from joblib import Parallel, delayed

from frites.core import MI_FUN, permute_mi_vector
from frites.stats import STAT_FUN
from frites.io import is_pandas_installed, is_xarray_installed, set_log_level
from frites.config import CONFIG

logger = logging.getLogger("frites")


class WfMi(object):
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

    def __init__(self, mi_type='cc', inference='rfx', verbose=None):
        """Init."""
        set_log_level(verbose)
        assert mi_type in ['cc', 'cd', 'ccd']
        assert inference in ['ffx', 'rfx']
        self._mi_type = mi_type
        self._inference = inference
        self.clean()

        logger.info(f"Workflow for computing mutual information ({mi_type}) "
                    f"and statistics ({inference}) has been defined")

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
        # track time and roi
        self._times, self._roi = dataset.times, dataset.roi_names


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

        self._mi, self._mi_p = mi, mi_p

        return mi, mi_p


    def _node_compute_stats(self, mi, mi_p, n_jobs=-1,
                            stat_method='rfx_cluster_ttest', **kw_stats):
        """Compute the non-parametric statistics.

        mi   = list of length n_roi composed with arrays of shape
               (n_subjects, n_times)
        mi_p = list of length n_roi composed with arrays of shape
               (n_perm, n_subjects, n_times)
        """
        # don't compute statistics
        if stat_method is None:
            return np.ones((len(mi), mi[0].shape[-1]), dtype=float)
        # get the function to evaluate statistics
        stat_fun = STAT_FUN[self._inference][stat_method]
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

    def _node_postprocessing(self, mi, pvalues, times, roi, mean_mi=True,
                             output_type='dataframe'):
        """Post preprocess outputs.

        This node take the mean mi across subjects and also enable to format
        outputs to NumPy, Pandas or Xarray.
        """
        # mean mi across subjects
        if mean_mi:
            logger.info("    Mean mi across subjects")
            mi = np.stack([k.mean(axis=0) for k in mi])
        # output type
        assert output_type in ['array', 'dataframe', 'dataarray', 'dataset']
        logger.info(f"    Formatting output type ({output_type})")
        force_np = not is_pandas_installed() and not is_xarray_installed()
        if force_np or (output_type is 'array'):       # numpy
            return mi, pvalues
        elif output_type is 'dataframe':               # pandas
            is_pandas_installed(raise_error=True)
            import pandas as pd
            mi = pd.DataFrame(mi.T, index=times, columns=roi)
            pvalues = pd.DataFrame(pvalues.T, index=times, columns=roi)
            return mi, pvalues
        elif output_type in ['dataarray', 'dataset']:  # xarray
            is_xarray_installed(raise_error=True)
            from xarray import DataArray, Dataset
            mi = DataArray(mi, dims=('roi', 'times'), coords=(roi, times))
            pvalues = DataArray(pvalues, dims=('roi', 'times'),
                                coords=(roi, times))
            if output_type is 'dataarray':
                return mi, pvalues
            return Dataset(dict(mi=mi, pvalues=pvalues))


    ###########################################################################
    #                             EXTERNALS
    ###########################################################################

    def fit(self, dataset, n_perm=1000, n_jobs=-1, output_type='dataframe',
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
                  with the Threshold Free Cluster Enhancement for cluster level
                  inference (see :func:`frites.stats.rfx_cluster_ttest_tfce`
                  :cite:`smith2009threshold`)
        output_type : {'array', 'dataframe', 'dataarray', 'dataset'}
            Convert the mutual information and p-values arrays either to
            pandas DataFrames (require pandas to be installed) either to a
            xarray DataArray or DataSet (require xarray to be installed)
        kw_stats : dict | {}
            Additional arguments to pass to the selected statistical method
            selected using the `stat_method` input parameter

        Returns
        -------
        mi, pvalues : array_like
            Array of mean mutual information across subjects and p-values of
            shape (n_roi, n_times) if `output_type` is 'array'. If
            `output_type` is 'dataframe', a pandas.DataFrame is returned. If
            `output_type` is 'dataarray' or 'dataset' a xarray.DataArray or
            xarray.Dataset are returned

        References
        ----------
        Maris and Oostenveld, 2007 :cite:`maris2007nonparametric`
        """
        # before performing any computations, we check if the statistical
        # method does exist
        try:
            if isinstance(stat_method, str):
                STAT_FUN[self._inference][stat_method]
            else:
                n_perm = 0
        except KeyError:
            m_names = [k.__name__ for k in STAT_FUN[self._inference].values()]
            raise KeyError(f"Selected statistical method `{stat_method}` "
                           f"doesn't exist. For {self._inference} inference, "
                           f"use either : {', '.join(m_names)}")
        # if mi and mi_p have already been computed, reuse it instead
        if len(self._mi) and len(self._mi_p):
            logger.info("    True and permuted mutual-information already "
                        "computed. Use WfMi.clean to reset "
                        "arguments")
            mi, mi_p = self._mi, self._mi_p
        else:
            self._node_prepare_data(dataset)
            mi, mi_p = self._node_compute_mi(dataset, n_perm=n_perm,
                                             n_jobs=n_jobs)
        # infer p-values
        pvalues = self._node_compute_stats(mi, mi_p, n_jobs=n_jobs,
                                           stat_method=stat_method, **kw_stats)
        # mean mi and format outputs
        outs = self._node_postprocessing(mi, pvalues, dataset.times,
                                         dataset.roi_names,
                                         output_type=output_type)

        return outs

    def clean(self):
        """Clean computations."""
        self._mi, self._mi_p = [], []

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
