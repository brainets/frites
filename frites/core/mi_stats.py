"""Utility functions for stat evaluation."""
import numpy as np


def permute_mi_vector(y, suj, mi_type='cc', inference='rfx', n_perm=1000):
    """Permute regressor variable for performing non-parameteric statistics.

    Parameters
    ----------
    y : array_like
        Array of shape (n_epochs,) to be permuted
    suj : array_like
        Array of shape (n_epochs,) used for permuting per subject
    mi_type : {'cc', 'cd', 'ccd'}
        Mutual information type
    inference : {'ffx', 'rfx'}
        Inference type (fixed or random effect)
    n_perm : int | 1000
        Number of permutations to return

    Returns
    -------
    y_p : list
        List of length (n_perm,) of random permutation of the regressor
    """
    y_p = []
    for p in range(n_perm):
        if inference == 'ffx':    # FFX (FIXED EFFECT)
            y_p += [np.random.permutation(y)]
        elif inference == 'rfx':  # RFX (RANDOM EFFECT)
            _y = y.copy()
            for s in np.unique(suj):
                # find everywhere the subject is present
                is_suj = suj == s
                # randomize per subject
                _y[is_suj] = np.random.permutation(y[is_suj])
            y_p += [_y]
    assert len(y_p) == n_perm

    return y_p
