"""Utility functions for stat evaluation."""
import numpy as np
import xarray as xr

from mne.utils import ProgressBar

from frites.utils import nonsorted_unique
from frites.dataset.ds_utils import multi_to_uni_conditions
from frites.io.io_attributes import check_attrs
from frites.io import logger, set_log_level


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


def dist_to_ci(dist, cis=[99], inference='ffx', rfx_es='mi', pop_mean=None):
    """Extract confidence bounds of a distribution.

    Parameters
    ----------
    dist : array_like
        Distribution of shape (n_boots, 1, n_times)
    cis : list | [99]
        List of confidence levels
    inference : {'ffx', 'rfx'}
        Statistical model of the group
    rfx_es : {'mi', 'tvalues'}
        RFX effect size type. Use either 'mi' (for MI in bits) or 'tvalues' if
        a t-test is required
    pop_mean : float | None
        Value to use for performing the t-test

    Returns
    -------
    cis : array_like
        Array describing the bounds of the confidence intervals. This array has
        a shape of (n_cis, 2, n_times)
    """
    assert inference in ['ffx', 'rfx']
    assert isinstance(cis, (list, tuple, np.ndarray))
    assert rfx_es in ['mi', 'tvalues']
    assert dist.ndim == 3

    # group level effect for the rfx
    if (inference == 'rfx') and (rfx_es == 'mi'):
        dist = dist.mean(1, keepdims=True)
    elif (inference == 'rfx') and (rfx_es == 'tvalues'):
        raise NotImplementedError()
        # assert isinstance(pop_mean, (int, float))
        # from frites.config import CONFIG
        # s_hat = CONFIG['TTEST_MNE_SIGMA']
        # sigma = s_hat * np.var(dist, axis=1, ddof=1).max()
        # dist = ttest_1samp(dist, pop_mean, axis=1, implementation='mne',
        #                    method='absolute', sigma=sigma)[:, np.newaxis, :]
    assert dist.shape[1] == 1  # (n_boots, 1, n_times)
    _, _, n_times = dist.shape

    # find bounds
    x_ci = np.zeros((len(cis), 2, n_times))
    for n_ci, ci in enumerate(cis):
        half_alpha = (100. - ci) / 2.
        x_ci[n_ci, 0, :] = np.percentile(dist, half_alpha, axis=0)
        x_ci[n_ci, 1, :] = np.percentile(dist, (100. - half_alpha), axis=0)

    return x_ci


def confidence_interval(data, axis=0, cis=95, n_boots=200, random_state=None,
                        fcn=None, skipna=True, verbose=None):
    """Compute the confidence interval of repeated measurements.

    Parameters
    ----------
    data : array_like
        Numpy array (or xarray.DataArray) of data
    axis : int | 0
        Axis along which to compute the confidence interval
    cis : int, list | 95
        Integer or list of confidence levels to extract. This input also
        supports computing standard deviation ('sd') and / or standard error on
        the mean ('sem')
    n_boots : int | 200
        Number of bootstraps to perform
    random_state : int | None
        Fix the random state of the machine (use it for reproducibility). If
        None, a random state is randomly assigned.
    fcn : function | None
        Summary statistics function. By default, the mean is used.
    skipna : bool | True
        Skip NaN when computing CI. By default NaN are skipped.

    Returns
    -------
    ci : array_like
        Array of confidence intervals of shape (n_ci, 2, ...) where n_ci
        refers to the number of desired confidence intervals (see input `cis`)
        and 2 refers to the lower and upper bounds.
    """
    set_log_level(verbose)
    # ---------------------------------- I/O ----------------------------------
    if isinstance(cis, (int, float, str)):
        cis = [cis]
    assert isinstance(cis, (list, tuple, np.ndarray))
    assert isinstance(n_boots, int)
    need_ci = np.any([isinstance(k, (int, float)) for k in cis])
    logger.info(f"    Estimating CI (cis={cis}, axis={axis}, "
                f"n_boots={n_boots}, skipna=True, "
                f"random_state={random_state})")

    # default functions
    if fcn is None:
        fcn = np.nanmean if skipna else np.mean
    fcn_std = np.nanstd if skipna else np.std

    # ------------------------------- DATAARRAY -------------------------------
    if isinstance(data, xr.DataArray):
        if isinstance(axis, str):
            axis = data.get_axis_num(axis)
        dims = [d for n_d, d in enumerate(data.dims) if n_d != axis]
        coords = [data[d].data for d in dims]
        attrs = data.attrs
        attrs.update(n_boots=n_boots, random_state=random_state,
                     skipna=skipna, fcn=fcn.__name__)
        attrs = check_attrs(attrs)
        name = 'CI' if data.name is None else data.name + '_CI'
        x = data.data
    else:
        x = data

    # ------------------------------- BOOSTRAPS -------------------------------
    if need_ci:
        # compute summary statistics
        part = bootstrap_partitions(x.shape[axis], n_partitions=n_boots,
                                    random_state=random_state)
        x_ci = []
        for k in range(n_boots):
            sl = [slice(None)] * x.ndim
            sl[axis] = part[k]
            x_ci.append(fcn(x[tuple(sl)], axis=axis))
        x_ci = np.stack(x_ci)

    # -------------------------------- CI / STD -------------------------------
    # infer ci bounds
    cib = []
    for n_ci, ci in enumerate(cis):
        if isinstance(ci, (int, float)):
            halpha = (100. - ci) / 2.
            _ci = np.percentile(x_ci, [halpha, 100. - halpha], axis=0)
        elif ci in ['sd', 'sem']:
            x_sd, x_m = fcn_std(x, axis=axis), fcn(x, axis=axis)
            if ci == 'sem':
                x_sd /= np.sqrt(x.shape[axis])
            _ci = np.stack([x_m - x_sd, x_m + x_sd])
        cib.append(_ci)
    cib = np.stack(cib)

    # --------------------------------- XARRAY --------------------------------
    # xarray formatting (if needed)
    if isinstance(data, xr.DataArray):
        cib = xr.DataArray(
            cib, dims=['ci', 'bound'] + dims,
            coords=[cis, ['low', 'high']] + coords,
            attrs=attrs, name=name
        )

    return cib


def trial_swap_surrogates(x, random_state=0, verbose=False):
    """Given the data, randomly swap the trials of the channels.

    Parameters
    ----------
    x: array_like
        Data array with dimensions ("trials", "roi", "time").
    random_state: int
        Seed used for the trial swapping

    Returns
    -------
    x_surr: array_like
        Data with randomized trials ("trials","roi","time").
    """
    set_log_level(verbose)

    # define the random state
    rnd = np.random.randint(1000) if not isinstance(
        random_state, int) else random_state
    rnd = np.random.RandomState(rnd)

    assert isinstance(x, (np.ndarray, xr.DataArray))

    # Get number of nodes and time points
    n_trials, n_nodes = x.shape[0], x.shape[1]

    # Surrogate data
    x_surr = np.zeros_like(x)
    # Array with trial indexes
    trials = np.arange(n_trials, dtype=int)

    # define progress bar
    pbar = ProgressBar(range(n_nodes), mesg='Trial shuffling')

    for c in range(n_nodes):
        # destroy roi-roi trial relation
        np.random.shuffle(trials)
        x_surr[:, c, :] = x[trials, c, :]
        pbar.update_with_increment_value(1)

    if isinstance(x, xr.DataArray):
        x_surr = xr.DataArray(
            x_surr, dims=x.dims, coords=x.coords, name=x.name, attrs=x.attrs
        )

    return x_surr


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from frites import set_mpl_style
    set_mpl_style()

    data = np.random.rand(5, 100)
    coords = [np.arange(k) for k in data.shape]
    dims = [str(k) for k in range(data.ndim)]
    data = xr.DataArray(
        data, dims=dims, coords=coords, attrs={'arg': 'test'}, name='Test'
    )
    ci = confidence_interval(data, axis=1, cis=[95, 99, 90, 'sd', 'sem'])
    # ci = confidence_interval(data, axis=1, cis='sem')

    if len(ci['ci']) > 1:
        fg = ci.plot(x='0', hue='bound', col='ci')
        for ax in np.ravel(fg.axes):
            plt.sca(ax)
            plt.plot(np.arange(data.shape[0]), data.data.mean(1), 'o')
    else:
        fg = ci.plot(x='0', hue='bound')
        plt.plot(np.arange(data.shape[0]), data.data.mean(1), 'o')
    plt.show()
