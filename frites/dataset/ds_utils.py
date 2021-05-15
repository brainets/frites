"""Utility functions for organizing dataset's data."""
import numpy as np

from frites.io import set_log_level, logger
from frites.config import CONFIG


def multi_to_uni_conditions(x, var_name=None, verbose=None):
    """Convert a discret vector that contains multiple conditions.

    This function can be used to convert a list of discret arrays, each
    reflecting possibly multivariate stimulus or conditions.

    Parameters
    ----------
    x : list
        List of multi-variate conditions. Each element of the list is an array
        of shape (n_trials, n_conditions), where the number of trials can
        varies across elements of the list but they all have to have the same
        number of conditions
    var_name : string | None
        The name of the variable (usefull for debugging)

    Returns
    -------
    x_new : list
        List of remapped conditions where each element of the list has a shape
        of (n_trials,)
    """
    set_log_level(verbose)
    # =============================== Checking ================================

    if not isinstance(x, (list, tuple)):
        return [x]
    assert all([type(x[0]) == type(k) for k in x])
    x_types = type(x[0])
    if not x_types == np.ndarray:
        return x
    # get if all variables are integers and multicolumns else skip it
    is_int = all([k.dtype in CONFIG['INT_DTYPE'] for k in x])
    is_ndim = all([k.ndim > 1 for k in x])
    if not is_int or not is_ndim:
        return x
    # test that all dimensions are equals
    same_dim = all([k.ndim == x[0].ndim for k in x])
    if not same_dim and isinstance(var_name, str):
        assert ValueError(f"Every array in the `{var_name}` input should "
                          "have the same number of dimensions")
    # otherwise find all possible pairs
    x_all = np.concatenate(x, axis=0)
    idx = np.unique(x_all, axis=0, return_index=True)[1]
    u_cat = x_all[sorted(idx), :]
    # show to the user the new categories
    user = []
    for n_c, cat in enumerate(u_cat):
        user += [f"{n_c}: [{', '.join([str(c) for c in cat])}]"]
    if isinstance(var_name, str):
        logger.debug(f"    The `{var_name}` input contains multiple conditions"
                     f" that have been remapped to : {'; '.join(user)}")
    # loop over subjects
    x_new = []
    for k in range(len(x)):
        x_cat = np.full((x[k].shape[0],), -1, dtype=int)
        for n_c, cat in enumerate(u_cat):
            x_cat[np.equal(x[k], cat.reshape(1, -1)).all(1)] = n_c
        assert x_cat.min() > -1, "Not all values have been replaced"
        x_new += [x_cat]

    return x_new


if __name__ == '__main__':
    y = [0, 0, 1, 1, 2, 1]
    z = [0, 1, 0, 0, 1, 2]
    # result = [0, 1, 1, 1, 2, 2]
    x = np.c_[y, z]
    x = None
    x_new = multi_to_uni_conditions([x], 'x', verbose='debug')
    print(x_new)
