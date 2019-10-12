"""Workflow for computing MI and evaluate statistics."""
import logging

import numpy as np
from joblib import Parallel, delayed

from frites.core import MI_FUN, permute_mi_vector
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


    def _node_compute_stats(self, mi, mi_p, n_jobs=-1, **kw_stats):
        pass

    ###########################################################################
    #                             EXTERNALS
    ###########################################################################

    def fit(self, dataset, n_perm=1000, n_jobs=-1, **kw_stats):
        """Run the workflow on a dataset.

        In order to run the worflow, you must first provide a dataset instance
        (see :class:`frites.dataset.DatasetEphy`)

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

        Returns
        -------
        """
        self._node_prepare_data(dataset)
        mi, mi_p = self._node_compute_mi(dataset, n_perm=n_perm, n_jobs=n_jobs)
        pvalues = self._node_compute_stats(mi, mi_p, n_jobs=n_jobs, **kw_stats)


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

    modality = 'intra'
    n_subjects = 3
    n_epochs = 100
    n_times = 100
    n_roi = 4
    n_sites_per_roi = 3
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
    WorkflowMiStats('cc', 'ffx').fit(dt, n_jobs=-1, n_perm=5)
