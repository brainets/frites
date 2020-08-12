"""Simulate local representation of mutual information."""
import numpy as np


"""
###############################################################################
                                 I(C; C)
###############################################################################
- MI between two continuous variables
- Single / Multi subjects simulations
"""


def sim_local_cc_ms(n_subjects, random_state=None, **kwargs):
    """Multi-subjects simulations for computing local MI (CC).

    This function can be used for simulating local representations of mutual
    information between two continuous variables (CC) across multiple subjects.

    Parameters
    ----------
    n_subjects : int
        Number of subjects
    kwargs : dict | {}
        Additional arguments are send to the function :func:`sim_local_cc_ss`

    Returns
    -------
    x : list
        List length n_subjects composed of data arrays each one with a shape of
        (n_epochs, n_channels, n_times)
    y : list
        List of length n_subjects composed of regressor arrays each one with a
        shape of (n_epochs)
    roi : list
        List of length n_subjects composed of roi arrays each one with a shape
        of (n_roi)
    times : array_like
        Time vector
    """
    # get the default cluster indices and covariance
    cl_index = kwargs.get('cl_index', [40, 60])
    cl_cov = kwargs.get('cl_cov', [.8])
    cl_sgn = kwargs.get('cl_sgn', [1])
    # repeat if not equal to subject length
    if len(cl_index) != n_subjects:
        cl_index = [cl_index] * n_subjects
    if len(cl_cov) != n_subjects:
        cl_cov = [cl_cov] * n_subjects
    if len(cl_sgn) != n_subjects:
        cl_sgn = cl_sgn * n_subjects
    if not isinstance(random_state, int):
        random_state = np.random.randint(1000)
    # now generate the data
    x, y, roi = [], [], []
    for n_s in range(n_subjects):
        # re-inject the indices and covariance
        kwargs['cl_index'] = cl_index[n_s]
        kwargs['cl_cov'] = cl_cov[n_s]
        kwargs['cl_sgn'] = cl_sgn[n_s]
        # generate the data of a single subject
        _x, _y, _roi, times = sim_local_cc_ss(random_state=random_state + n_s,
                                              **kwargs)
        # merge data
        x += [_x]
        y += [_y]
        roi += [_roi]

    return x, y, roi, times


def sim_local_cc_ss(n_epochs=10, n_times=100, n_roi=1, cl_index=[40, 60],
                    cl_cov=[.8], cl_sgn=1, random_state=None):
    """Single-subject simulations for computing local MI (CC).

    This function can be used for simulating local representations of mutual
    information between two continuous variables (CC) for a single subject.

    Parameters
    ----------
    n_epochs : int | 30
        Number of trials
    n_times : int | 100
        Number of time points
    n_roi : int | 1
        Number of ROI
    cl_index : array_like | [40, 60]
        Sample indices where the clusters are located. Should be an array of
        shape (n_clusters, 2)
    cl_cov : array_like | [.8]
        Covariance level between the data and the regressor variable. Should be
        an array of shape (n_clusters,)
    cl_sgn : {-1, 1}
        Sign of the correlation. Use -1 for anti-correlated variables and 1
        for correlated variables
    random_state : int | None
        Random state (use it for reproducibility)

    Returns
    -------
    x : array_like
        Data array of shape (n_epochs, n_channels, n_times)
    y : array_like
        Regressor array of shape (n_epochs,)
    roi : array_like
        Array of ROI names of shape (n_roi,)
    times : array_like
        Time vector of shape (n_times,)
    """
    random_state = np.random.randint(100) if not isinstance(
        random_state, int) else random_state
    rnd = np.random.RandomState(random_state)

    # -------------------------------------------------------------------------
    # Pick random n_roi
    roi = np.array([f"roi_{k}" for k in range(n_roi)])

    # -------------------------------------------------------------------------
    # check cluster types
    cl_index, cl_cov = np.atleast_2d(cl_index), np.asarray(cl_cov)
    assert (cl_index.shape[-1] == 2) and (cl_cov.ndim == 1)
    assert cl_sgn in [-1, 1]
    if cl_index.shape[0] == 1:
        cl_index = np.tile(cl_index, (n_roi, 1))
    if cl_cov.shape[0] == 1:
        cl_cov = np.repeat(cl_cov, n_roi)
    assert (cl_index.shape == (n_roi, 2)) and (cl_cov.shape == (n_roi,))

    # -------------------------------------------------------------------------
    # Built a random dataset
    x = rnd.randn(n_epochs, n_roi, n_times)
    y = rnd.randn(n_epochs).reshape(-1, 1)

    # -------------------------------------------------------------------------
    # Introduce a correlation between the data and the regressor
    for num, (idx, cov) in enumerate(zip(cl_index, cl_cov)):
        if not np.isfinite(cov): continue  # noqa
        # define correlation strength
        t_len = idx[1] - idx[0]
        epsilon = np.sqrt((1. - cov ** 2) / cov ** 2)
        # Generate noise
        rnd_noise = np.random.RandomState(random_state + num + 1)
        noise = epsilon * rnd_noise.randn(n_epochs, t_len)
        x[:, num, idx[0]:idx[1]] = cl_sgn * y + noise
    times = np.arange(n_times)

    return x, y.ravel(), roi, times


"""
###############################################################################
                                 I(C; D)
###############################################################################
- MI between a continuous and a discret variable
- Single / Multi subjects simulations
"""


def sim_local_cd_ms(n_subjects, **kwargs):
    """Multi-subjects simulations for computing local MI (CD).

    This function can be used for simulating local representations of mutual
    information between a continuous and a discret variables (CD) for a single
    subject.

    Parameters
    ----------
    n_subjects : int
        Number of subjects
    kwargs : dict | {}
        Additional arguments are send to the function :func:`sim_local_cd_ss`

    Returns
    -------
    x : list
        List length n_subjects composed of data arrays each one with a shape of
        (n_epochs, n_channels, n_times)
    y : list
        List of length n_subjects composed of regressor arrays each one with a
        shape of (n_epochs)
    roi : list
        List of length n_subjects composed of roi arrays each one with a shape
        of (n_roi)
    times : array_like
        Time vector of shape (n_times,)
    """
    # get the default cluster indices and covariance
    cl_index = kwargs.get('cl_index', [40, 60])
    cl_cov = kwargs.get('cl_cov', [.8])
    # repeat if not equal to subject length
    if len(cl_index) != n_subjects:
        cl_index = [cl_index] * n_subjects
    if len(cl_cov) != n_subjects:
        cl_cov = [cl_cov] * n_subjects
    # now generate the data
    x, y, roi = [], [], []
    for n_s in range(n_subjects):
        # re-inject the indices and covariance
        kwargs['cl_index'] = cl_index[n_s]
        kwargs['cl_cov'] = cl_cov[n_s]
        # generate the data of a single subject
        _x, _y, _roi, times = sim_local_cd_ss(random_state=n_s, **kwargs)
        # merge data
        x += [_x]
        y += [_y]
        roi += [_roi]

    return x, y, roi, times


def sim_local_cd_ss(n_conditions=2, n_epochs=10, n_times=100, n_roi=1,
                    cl_index=[40, 60], cl_cov=[.8], random_state=None):
    """Single-subject simulations for computing local MI (CD).

    This function can be used for simulating local representations of mutual
    information between a continuous and a discret variable (CD) for a single
    subject.

    Parameters
    ----------
    n_conditions : int | 2
        Number of conditions
    n_epochs : int | 30
        Number of trials
    n_times : int | 100
        Number of time points
    n_roi : int | 1
        Number of ROI
    cl_index : array_like | [40, 60]
        Sample indices where the clusters are located. Should be an array of
        shape (n_clusters, 2)
    cl_cov : array_like | [.8]
        Covariance level between the data and the regressor variable. Should be
        an array of shape (n_clusters,)
    random_state : int | None
        Random state (use it for reproducibility)

    Returns
    -------
    x : array_like
        Data array of shape (n_epochs, n_channels, n_times)
    y : array_like
        Condition array of shape (n_epochs,)
    roi : array_like
        Array of ROI names of shape (n_roi,)
    times : array_like
        Time vector of shape (n_times,)
    """
    random_state = np.random.randint(100) if not isinstance(
        random_state, int) else random_state
    rnd = np.random.RandomState(random_state)

    # -------------------------------------------------------------------------
    # Pick random n_roi
    roi = np.array([f"roi_{k}" for k in range(n_roi)])

    # -------------------------------------------------------------------------
    # check cluster types
    cl_index, cl_cov = np.atleast_2d(cl_index), np.asarray(cl_cov)
    assert (cl_index.shape[-1] == 2) and (cl_cov.ndim == 1)
    if cl_index.shape[0] == 1:
        cl_index = np.tile(cl_index, (n_roi, 1))
    if cl_cov.shape[0] == 1:
        cl_cov = np.repeat(cl_cov, n_roi)
    assert (cl_index.shape == (n_roi, 2)) and (cl_cov.shape == (n_roi,))

    # -------------------------------------------------------------------------
    # linearly spaced values taken from a gaussian distribution
    res = 100
    pick_up = np.linspace(0, res - 1, n_conditions).astype(int)
    values = np.sort(rnd.randn(res))[pick_up]
    # regressor variable
    y_regr = np.repeat(values, np.round(n_epochs, n_conditions))
    y_regr = rnd.permutation(y_regr)[0:n_epochs]
    # condition variable
    _, y = np.unique(y_regr, return_inverse=True)
    y_regr = y_regr.reshape(-1, 1)
    x = rnd.randn(n_epochs, n_roi, n_times)

    # -------------------------------------------------------------------------
    # Introduce a correlation between the data and the regressor
    for num, (idx, cov) in enumerate(zip(cl_index, cl_cov)):
        # define correlation strength
        t_len = idx[1] - idx[0]
        epsilon = np.sqrt((1. - cov ** 2) / cov ** 2)
        # Generate noise
        rnd_noise = np.random.RandomState(random_state + num + 1)
        noise = epsilon * rnd_noise.randn(n_epochs, t_len)
        x[:, num, idx[0]:idx[1]] = y_regr + noise
    times = np.arange(n_times)

    return x, y.astype(int), roi, times


"""
###############################################################################
                                 I(C; C | D)
###############################################################################
- MI between two continuous variables conditioned by a discret variable
- Single / Multi subjects simulations
"""


def sim_local_ccd_ms(n_subjects, **kwargs):
    """Multi-subjects simulations for computing local MI (CCD).

    This function can be used for simulating local representations of mutual
    information between two continuous variables conditioned by a third discret
    one (CCD) across multiple subjects.

    Parameters
    ----------
    n_subjects : int
        Number of subjects
    kwargs : dict | {}
        Additional arguments are send to the function :func:`sim_local_ccd_ss`

    Returns
    -------
    x : list
        List length n_subjects composed of data arrays each one with a shape of
        (n_epochs, n_channels, n_times)
    y : list
        List of length n_subjects composed of regressor arrays each one with a
        shape of (n_epochs)
    z : array_like
        Condition array of shape (n_epochs,)
    roi : list
        List of length n_subjects composed of roi arrays each one with a shape
        of (n_roi)
    times : array_like
        Time vector
    """
    n_c = kwargs.get('n_conditions', 2)
    if 'n_conditions' in list(kwargs.keys()):
        kwargs.pop('n_conditions')
    x, y, roi, times = sim_local_cc_ms(n_subjects, **kwargs)
    n_e = len(y[0])
    z = [np.random.randint(0, n_c, (n_e,)) for k in range(n_subjects)]

    return x, y, z, roi, times


def sim_local_ccd_ss(n_epochs=10, n_times=100, n_roi=1, n_conditions=2,
                     cl_index=[40, 60], cl_cov=[.8], random_state=None):
    """Single-subject simulations for computing local MI (CC).

    This function can be used for simulating local representations of mutual
    information between two continuous variables conditioned by a third discret
    one (CCD) for a single subject.

    Parameters
    ----------
    n_epochs : int | 30
        Number of trials
    n_times : int | 100
        Number of time points
    n_roi : int | 1
        Number of ROI
    n_conditions : int | 2
        Number of conditions
    cl_index : array_like | [40, 60]
        Sample indices where the clusters are located. Should be an array of
        shape (n_clusters, 2)
    cl_cov : array_like | [.8]
        Covariance level between the data and the regressor variable. Should be
        an array of shape (n_clusters,)
    random_state : int | None
        Random state (use it for reproducibility)

    Returns
    -------
    x : array_like
        Data array of shape (n_epochs, n_channels, n_times)
    y : array_like
        Regressor array of shape (n_epochs,)
    z : array_like
        Condition array of shape (n_epochs,)
    roi : array_like
        Array of ROI names of shape (n_roi,)
    times : array_like
        Time vector of shape (n_times,)
    """
    x, y, roi, times = sim_local_cc_ss(
        n_epochs=n_epochs, n_times=n_times, n_roi=n_roi, cl_index=cl_index,
        cl_cov=cl_cov, random_state=random_state)
    z = np.random.randint(0, n_conditions, (n_epochs,))
    return x, y, z, roi, times
