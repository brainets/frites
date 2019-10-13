"""Simulate datasets for computing MI (cc / cd / ccd)."""
import numpy as np

from frites.config import CONFIG


def sim_mi_cc(x, snr=.9):
    """Extract a continuous variable from data.

    This function can be used to then evaluate the mutual information between
    some neurophysiological data and a continuous variable (e.g regressor,
    model based regressions etc.).

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
    n_times, n_epochs = x[0].shape[-1], x[0].shape[0]
    # cluster definition (10% length around central point)
    middle = int(np.round(n_times / 2))
    width = int(np.round(n_times * .2 / 2))
    cluster = slice(middle - width, middle + width)
    # ground truth definition
    gt = np.zeros((n_times,), dtype=bool)
    gt[cluster] = True
    # find mean and deviation of the regressors
    _y = [k[..., cluster].mean(axis=(1, 2)) for k in x]
    cat_y = np.r_[tuple(_y)]
    loc, scale = np.mean(cat_y), np.std(cat_y)
    # generate random noise
    _noise = [k * np.random.normal(loc, scale, size=(n_epochs,)) for k in _y]
    y = [snr * k + (1. - snr) * i for k, i in zip(_y, _noise)]
    return y, gt


def sim_mi_cd(x, n_conditions=3, snr=.9):
    """Extract a discret variable from data.

    This function can be used to then evaluate the mutual information between
    some neurophysiological data and a discret variable (e.g conditions).

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
    # cluster definition (10% length around central point)
    middle = int(np.round(n_times / 2))
    width = int(np.round(n_times * .2 / 2))
    cluster = slice(middle - width, middle + width)
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


def sim_mi_ccd():
    """Soon."""
    raise NotImplementedError("TODO")


if __name__ == '__main__':
    from frites.simulations import sim_multi_suj_ephy
    from frites.core import mi_nd_gg, mi_model_nd_gd
    import matplotlib.pyplot as plt

    modality = 'intra'
    n_subjects = 1
    n_epochs = 100
    n_times = 50
    n_roi = 1
    n_sites_per_roi = 1
    as_mne = True
    x, roi, time = sim_multi_suj_ephy(n_subjects=n_subjects, n_epochs=n_epochs,
                                      n_times=n_times, n_roi=n_roi,
                                      n_sites_per_roi=n_sites_per_roi,
                                      as_mne=as_mne, modality=modality,
                                      random_state=1)

    plt.subplot(211)
    for snr in np.linspace(.1, 1., 10, endpoint=True):
        x, y, gt = sim_mi_cd(x, snr=snr)
        x_0 = x[0].squeeze()
        if y[0].dtype == int:
            mi = mi_model_nd_gd(x_0, y[0], traxis=0)
        else:
            y_0 = np.tile(y[0].reshape(-1, 1), (1, n_times))
            mi = mi_nd_gg(x_0, y_0, traxis=0)

        plt.plot(mi, label=str(snr))
    plt.legend()
    plt.subplot(212)
    plt.plot(gt)
    plt.show()
