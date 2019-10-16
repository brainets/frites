"""Simulate datasets for computing MI (cc / cd / ccd)."""
import numpy as np

from frites.config import CONFIG


def _get_cluster(n_times, location='center', perc=.2):
    """Get a cluster that will exhibit an increase in mutual information.

    Parameters
    ----------
    n_times : int
        Number of time points
    location : {'left', 'center', 'right'}
        Location of the cluster
    perc : float | .2
        Length of the cluster (fraction of the number of time points)

    Returns
    -------
    cluster : slice
        Slice object where the cluster is located
    """
    middle = int(np.round(n_times / 2))
    width = int(np.round(n_times * perc / 2))
    if location == 'left':
        cluster = slice(width, 2 * width)
    elif location == 'center':
        cluster = slice(middle - width, middle + width)
    elif location == 'right':
        cluster = slice(-2 * width, -width)
    return cluster


def sim_mi_cc(x, snr=.9):
    """Extract a continuous variable from data.

    This function can be used to then evaluate the mutual information between
    some neurophysiological data and a continuous variable (e.g regressor,
    model based regressions etc.).

    .. math::

        I(C; C)

    This function takes as an input some random or real data and generates a
    continuous variable from it. If you want to generate some compatible
    random data see the function :func:`sim_multi_suj_ephy`.

    Parameters
    ----------
    x : list
        List of data coming from multiple subjects. Each element of this must
        be an array of shape (n_epochs, n_sites, n_times)
    snr : float | 80.
        Signal to noise ratio between [0, 1] (0 = no noise ; 1 = pure noise)

    Returns
    -------
    y : list
        List of length (n_subjects,) of continuous variables. Each array has a
        shape of (n_epochs,)
    gt : array_like
        Ground truth array of length (n_times,). This boolean array contains
        True where a cluster has been defined

    See also
    --------
    sim_multi_suj_ephy
    sim_mi_cd
    sim_mi_ccd
    """
    assert 0 < snr <= 1.
    assert isinstance(x, list)
    # if mne types, turn into arrays
    if isinstance(x[0], CONFIG["MNE_EPOCHS_TYPE"]):
        x = [x[k].get_data() for k in range(len(x))]
    n_times = x[0].shape[-1]
    # cluster definition (20% length around central point)
    cluster = _get_cluster(n_times, location='center', perc=.2)
    # ground truth definition
    gt = np.zeros((n_times,), dtype=bool)
    gt[cluster] = True
    # find mean and deviation of the regressors
    _y = [k[..., cluster].mean(axis=(1, 2)) for k in x]
    cat_y = np.r_[tuple(_y)]
    loc, scale = np.mean(cat_y), np.std(cat_y)
    # generate random noise
    _noise = [k * np.random.normal(loc, scale, size=(len(k),)) for k in _y]
    y = [snr * k + (1. - snr) * i for k, i in zip(_y, _noise)]
    return y, gt


def sim_mi_cd(x, n_conditions=3, snr=.9):
    """Extract a discret variable from data.

    This function can be used to then evaluate the mutual information between
    some neurophysiological data and a discret variable (e.g conditions).

    .. math::

        I(C; D)

    This function takes as an input some random or real data and generates a
    discret variable from it. If you want to generate some compatible
    random data see the function :func:`sim_multi_suj_ephy`.

    Parameters
    ----------
    x : list
        List of data coming from multiple subjects. Each element of this must
        be an array of shape (n_epochs, n_sites, n_times)
    n_conditions : int | 3
        Number of conditions to extract in the discret variable
    snr : float | 80.
        Signal to noise ratio between [0, 1] (0 = no noise ; 1 = pure noise)

    Returns
    -------
    x : list
        List of modified data
    y : list
        List of length (n_subjects,) of discret variables. Each array has a
        shape of (n_epochs,)
    gt : array_like
        Ground truth array of length (n_times,). This boolean array contains
        True where a cluster has been defined

    See also
    --------
    sim_multi_suj_ephy
    sim_mi_cc
    sim_mi_ccd
    """
    assert 0 < snr <= 1.
    assert isinstance(x, list)
    # if mne types, turn into arrays
    if isinstance(x[0], CONFIG["MNE_EPOCHS_TYPE"]):
        x = [x[k].get_data() for k in range(len(x))]
    n_times, n_epochs = x[0].shape[-1], x[0].shape[0]
    # cluster definition (20% length around central point)
    cluster = _get_cluster(n_times, location='center', perc=.2)
    # ground truth definition
    gt = np.zeros((n_times,), dtype=bool)
    gt[cluster] = True
    # find mean and deviation of the regressors
    y = []
    for s in range(len(x)):
        # sort clusters according to size trials
        rows = x[s][..., cluster].mean(axis=2).argsort(0)
        cols = np.arange(rows.shape[1]).reshape(1, -1)
        x[s] = x[s][rows, cols, :]  # dope broadcast indexing bro :D
        # define the condition variable
        _y = np.linspace(0, n_conditions, n_epochs, endpoint=False)
        _y = np.floor(_y).astype(int)
        # define the number of trials to shuffle
        n_to_shuffle = np.round(n_epochs * (1. - snr)).astype(int)
        # define the swaping indices and perform swaping
        swap_1 = np.random.permutation(np.arange(n_epochs))[:n_to_shuffle]
        swap_2 = np.random.permutation(np.arange(n_epochs))[:n_to_shuffle]
        _y[swap_1] = _y[swap_2]
        y += [_y]

    return x, y, gt


def sim_mi_ccd(x, snr=.9):
    """Extract a continuous and a discret variable from data.

    This function can be used to then evaluate the mutual information between
    some neurophysiological data and a regressor, conditioned by a discret
    variable.

    .. math::

        I(C; C | D)

    This function takes as an input some random or real data and generates a
    continuous and a discret variable from it. If you want to generate some
    compatible random data see the function :func:`sim_multi_suj_ephy`.

    Parameters
    ----------
    x : list
        List of data coming from multiple subjects. Each element of this must
        be an array of shape (n_epochs, n_sites, n_times)
    snr : float | 80.
        Signal to noise ratio between [0, 1] (0 = no noise ; 1 = pure noise)

    Returns
    -------
    y : list
        List of length (n_subjects,) of continuous variables. Each array has a
        shape of (n_epochs,)
    z : list
        List of length (n_subjects,) of discret variables. Each array has a
        shape of (n_epochs,)
    gt : array_like
        Ground truth array of length (n_times,). This boolean array contains
        True where a cluster has been defined

    See also
    --------
    sim_multi_suj_ephy
    sim_mi_cc
    sim_mi_cd
    """
    assert 0 < snr <= 1.
    assert isinstance(x, list)
    # if mne types, turn into arrays
    if isinstance(x[0], CONFIG["MNE_EPOCHS_TYPE"]):
        x = [x[k].get_data() for k in range(len(x))]
    n_times, n_epochs = x[0].shape[-1], x[0].shape[0]
    gp_1, gp_2 = np.array_split(np.arange(n_epochs), 2)
    z = [np.array([0] * len(gp_1) + [1] * len(gp_2))] * len(x)
    # cluster definition (20% length around central point)
    cl_left = _get_cluster(n_times, location='left', perc=.2)
    cl_right = _get_cluster(n_times, location='right', perc=.2)
    # ground truth definition
    gt = np.zeros((n_times,), dtype=bool)
    gt[cl_left] = True
    gt[cl_right] = True
    # find mean and deviation of the regressors
    # n_epochs, n_sites, n_times
    _y_left = [k[gp_1, :, cl_left].mean(axis=(1, 2)) for k in x]
    _y_right = [k[gp_2, :, cl_right].mean(axis=(1, 2)) for k in x]
    _y = [np.r_[k, i] for k, i in zip(_y_left, _y_right)]
    cat_y = np.r_[tuple(_y)]
    loc, scale = np.mean(cat_y), np.std(cat_y)
    # generate random noise
    _noise = [k * np.random.normal(loc, scale, size=(len(k),)) for k in _y]
    y = [snr * k + (1. - snr) * i for k, i in zip(_y, _noise)]
    return y, z, gt
