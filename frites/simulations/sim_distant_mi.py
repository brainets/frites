"""Simulate distant mi."""
import numpy as np
from scipy.stats import norm


def sim_distant_cc_ms(n_subjects, **kwargs):
    """Multi-subjects simulations for computing connectivity (CC).

    This function can be used to test and check the FIT.

    Parameters
    ----------
    n_subjects : int
        Number of subjects
    kwargs : dict | {}
        Additional arguments are send to the function :func:`sim_distant_cc_ss`

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
    x, y, roi = [], [], []
    for s in range(n_subjects):
        _x, _y, _roi = sim_distant_cc_ss(**kwargs)
        x += [_x]
        y += [_y]
        roi += [_roi]
    times = np.linspace(-1, 3, _x.shape[-1])

    return x, y, roi, times


def sim_distant_cc_ss(n_epochs=200, n_roi=2, random_state=None):
    """Single-subject simulation for computing connectivity (CC).

    This function can be used to test and check the FIT.

    Parameters
    ----------
    n_epochs : int | 30
        Number of trials
    n_roi : int | 1
        Number of ROI
    random_state : int | None
        Random state (use it for reproducibility)

    Returns
    -------
    x : array_like
        Data of shape (n_roi, n_epochs, n_pts)
    y : array_like
        Array of stimulus of shape (n_epochs,)
    roi : array_like
        Array of roi names
    """
    x_s, x_t, y = sim_gauss_fit(n_epochs=n_epochs, random_state=random_state)
    x = np.stack((x_s, x_t), axis=1)
    roi = np.array([f"roi_{k}" for k in range(n_roi)])
    return x, y, roi


def sim_gauss_fit(stim_type='cont_linear', n_epochs=400, n_sti=4, n_pts=400,
                  stim_amp=10, info_loss=50, stim_onset_x=200,
                  stim_onset_y=240, random_state=None):
    """Simulated signals for the FIT.

    Parameters
    ----------
    stim_type : {'cont_linear', 'discrete_stim', 'cont_flat'}
        Stimulation type
    n_epochs : int | 4000
        Number of trials
    n_sti : int | 4
        Number of stimulis
    n_pts : int | 400
        Number of time points
    stim_amp : int | 10
        Stimulus amplitude
    info_loss : int | 50
        Information loss between x and y
    stim_onset_x : int | 200
        Stimulus onset for the x signal
    stim_onset_y : int | 240
        Stimulus onset for the y signal

    Returns
    -------
    x, y : array_like
        Signals of shape (n_epochs, n_pts)
    stim : array_like
        Stimulus of shape (n_epochs,)
    """
    random_state = np.random.randint(100) if not isinstance(
        random_state, int) else random_state
    rnd = np.random.RandomState(random_state)
    assert stim_type in ['discrete_stim', 'cont_linear', 'cont_flat']
    if stim_type == 'discrete_stim':
        stim = np.sort(rnd.randint(n_sti, n_epochs, 1) - 1)
    elif stim_type == 'cont_linear':
        stim = (np.arange(n_epochs) / (n_epochs - 1)) * n_sti
    elif stim_type == 'cont_flat':
        stim = np.ones((1, n_epochs))
        stim += rnd.rand(len(stim)) / 100.
    # gaussian profile for stimulus amplitude
    vec = norm.pdf(np.arange(-5, 5, .25), 0, 2)
    gauss_stim = vec - vec.min()
    # normalise Gaussian profile to 1
    gauss_stim /= gauss_stim.max()
    # delay of information transfer
    delay = stim_onset_y - stim_onset_x
    # stimulus offset in x and y
    stim_offset_x = stim_onset_x + len(gauss_stim) - 1
    # stim_offset_y = stim_onset_y + len(gauss_stim) - 1
    # init signals x and y
    x = np.ones((n_pts + delay, n_epochs))
    y = np.ones((n_pts, n_epochs))
    # xn = np.ones((n_pts , n_epochs))
    # add the Gaussian profile
    sl = slice(stim_onset_x + delay, stim_offset_x + delay + 1)
    x[sl, :] += gauss_stim.reshape(-1, 1) * stim * stim_amp
    # add Poisson noise
    rnd_x = np.random.RandomState(random_state + 1)
    x = rnd_x.poisson(x)
    # create y signal
    y += x[:n_pts, :] / info_loss * stim_amp
    # reshape X
    x = x[delay:, :]
    # add noise on y
    rnd_y = np.random.RandomState(random_state + 2)
    y = rnd_y.poisson(y)
    # add random noise to remove equal values for gcmi
    x = x + rnd_x.rand(*x.shape) / 100.
    y = y + rnd_y.rand(*x.shape) / 100.
    return x.T, y.T, stim