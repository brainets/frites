"""Utility functions for stat evaluation."""
import numpy as np


def permute_mi_vector(y, suj, mi_type='cc', inference='rfx', n_perm=1000,
                      random_state=None):
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
    random_state : int | None
        Fix the random state of the machine (use it for reproducibility). If
        None, a random state is randomly assigned.

    Returns
    -------
    y_p : list
        List of length (n_perm,) of random permutation of the regressor
    """
    # fix the random starting point
    rnd_start = np.random.randint(1000) if not isinstance(
        random_state, int) else random_state

    y_p = []
    for p in range(n_perm):
        rnd = np.random.RandomState(rnd_start + p)
        if inference == 'ffx':    # FFX (FIXED EFFECT)
            # subject-wise randomization
            y_p += [rnd.permutation(y)]
        elif inference == 'rfx':  # RFX (RANDOM EFFECT)
            _y = y.copy()
            for s in np.unique(suj):
                # find everywhere the subject is present
                is_suj = suj == s
                # randomize per subject
                _y[is_suj] = rnd.permutation(y[is_suj])
            y_p += [_y]
    assert len(y_p) == n_perm

    return y_p


def permute_mi_trials(suj, inference='rfx', n_perm=1000, random_state=None):
    """Generate random partitions for swapping trials.

    Parameters
    ----------
    suj : array_like
        Array of shape (n_epochs,) used for permuting per subject
    inference : {'ffx', 'rfx'}
        Inference type (fixed or random effect)
    n_perm : int | 1000
        Number of permutations to return
    random_state : int | None
        Fix the random state of the machine (use it for reproducibility). If
        None, a random state is randomly assigned.

    Returns
    -------
    y_p : list
        List of length (n_perm,) of random partitions for permuting trials
    """
    # fix the random starting point
    rnd_start = np.random.randint(1000) if not isinstance(
        random_state, int) else random_state
    n_trials = len(suj)

    y_p = []
    for p in range(n_perm):
        rnd = np.random.RandomState(rnd_start + p)
        y = np.arange(n_trials)
        if inference == 'ffx':    # FFX (FIXED EFFECT)
            # subject-wise randomization
            y_p += [rnd.permutation(y)]
        elif inference == 'rfx':  # RFX (RANDOM EFFECT)
            _y = y.copy()
            for s in np.unique(suj):
                # find everywhere the subject is present
                is_suj = suj == s
                # randomize per subject
                _y[is_suj] = rnd.permutation(y[is_suj])
            y_p += [_y]
    assert len(y_p) == n_perm

    return y_p
