"""Combine two WfMi workflows."""
import numpy as np

from frites.io import set_log_level, logger


class WfMiCombine(object):
    """Combine outputs of two WfMi workflows.

    This class can be used to combine two workflows of WfMi into a single
    one. The combination is performed by taking the effect size and
    permutations of both workflows and substract them (i.e. wf_1 - wf_2)

    An typical usecase could be :

        * Fit a first workflow wf_1 for the difference between two stimulus
        * Fit a second workflow wf_2 for the difference between two other
          stimulus
        * Combine the two workflows (wf_1 - wf_2)
        * Perform the statistics to investigate whether mi_1 > mi_2

    Parameters
    ----------
    wf_1, wf_2 : frites.workflow.WfMi
        Two workflows of WfMi. Important, but the two workflows have to be
        fitted individually first.

    Returns
    -------
    wf : frites.workflow.WfMi
        A new workflow with combined data
    """

    def __init__(self, wf_1, wf_2, verbose=None):
        """Init."""
        pass

    def __new__(self, wf_1, wf_2, verbose=None):
        set_log_level(verbose)
        logger.info('Combining the data of two workflows')

        # ______________________________ CHECKING _____________________________
        # check that parameters are equals
        logger.info('    Check that parameters are equals between workflows')
        for p in ['mi_type', 'inference', 'n_perm', 'kernel']:
            p_1, p_2 = wf_1.attrs[p], wf_2.attrs[p]
            assert p_1 == p_2, (
                f"Parameter {p} not the same between workflows "
                f"(wf_1={p_1} != wf_2={p_2})")

        # check that mi and permutations have been computed
        assert (len(wf_1._mi) >= 1) and (len(wf_2._mi) >= 1), (
            "Both workflows have to be fitted first")
        assert (len(wf_1._mi_p) >= 1) and (len(wf_2._mi_p) >= 1), (
            "Both workflows have to be fitted first")

        # check that the length are equals
        np.testing.assert_array_equal(wf_1._roi, wf_2._roi)
        np.testing.assert_array_equal(wf_1._times, wf_2._times)
        np.testing.assert_array_equal(wf_1._mi_dims, wf_2._mi_dims)
        c_1, c_2 = wf_1._mi_coords, wf_2._mi_coords
        for (k_1, v_1), (k_2, v_2) in zip(c_1.items(), c_2.items()):
            np.testing.assert_array_equal(k_1, k_2)
            np.testing.assert_array_equal(v_1, v_2)
        assert len(wf_1._mi) == len(wf_2._mi)
        assert len(wf_1._mi_p) == len(wf_1._mi_p)

        # _______________________________ COMBINE _____________________________
        logger.info('    Combining workflows')

        # deepcopy the workflow and replace mi and permutations
        wf = wf_1.copy()
        wf._mi = [k - i for k, i in zip(wf_1._mi, wf_2._mi)]
        wf._mi_p = [k - i for k, i in zip(wf_1._mi_p, wf_2._mi_p)]

        return wf
