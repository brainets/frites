"""Utility functions for stat evaluation."""
import numpy as np

from frites.utils import nonsorted_unique
from frites.dataset.ds_utils import multi_to_uni_conditions


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


def bootstrap_partitions(n_epochs, *groups, n_partitions=200,
                         random_state=None):
    """Generate partitions for bootstrap.

    Parameters
    ----------
    n_epochs : int
        Number of epochs
    groups : array_like
        Groups within which permutations are performed. Should be arrays of
        shape (n_epochs,) and of type int
    n_partitions : int | 200
        Number of partitions to get
    random_state : int | None
        Fix the random state of the machine (use it for reproducibility). If
        None, a random state is randomly assigned.

    Returns
    -------
    partitions : list
        List of arrays describing the partitions within groups or not
    """
    from sklearn.utils import resample

    # define the random state
    rnd = np.random.randint(1000) if not isinstance(
        random_state, int) else random_state

    # manage groups
    if not len(groups):
        groups = np.zeros((n_epochs), dtype=int)
    else:
        if len(groups) == 1:
            groups = groups[0]
        else:
            groups = multi_to_uni_conditions(
                [np.stack(groups, axis=1)], var_name='boot', verbose=False)[0]
    u_groups = nonsorted_unique(groups)

    # generate the partitions
    partitions = []
    for n_p in range(n_partitions):
        _part = np.arange(n_epochs)
        for n_g in u_groups:
            is_group = groups == n_g
            n_group = is_group.sum()
            _part[is_group] = resample(
                _part[is_group], n_samples=n_group, random_state=rnd + n_p)
        partitions.append(_part)

    return partitions
