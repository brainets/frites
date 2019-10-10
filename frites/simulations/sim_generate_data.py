"""Generate some random data."""
import numpy as np
from scipy.stats import zscore
from scipy.signal import savgol_filter
from itertools import product

from frites.config import CONFIG


def sim_single_suj_ephy(modality="meeg", sf=512., n_times=1000, n_roi=1,
                        n_sites_per_roi=1, n_epochs=100, n_sines=100, f_min=.5,
                        f_max=160., noise=10, as_mne=False, random_state=None):
    """Simulate electrophysiological data of a single subject.

    This function generate some illustrative random electrophysiological data
    using sum of sines.

    Parameters
    ----------
    modality : {"meeg", "intra"}
        Recording modality. Can either be "meeg" (MEG / EEG) or "intra"
        (sEEG / ECoG)
    sf : float | 512.
        The sampling frequency
    n_times : int | 1000
        The number of time points.
    n_sites_per_roi : int | 1
        Number of recording sites (or sensors) per region of interest.
    n_roi : int | 1
        Number of region of interest (ROI).
    n_epochs : int | 100
        Number of trials
    n_sines : int | 100
        Number of sines composing each epoch.
    f_min : float | .5
        Minimum frequency for sines.
    f_max : float | 160.
        Maximum frequency for sines.
    noise : float | 10.
        Noise level.
    as_mne : bool | False
        If True, data are converted to a mne.EpochsArray structure
    random_state : int | None
        Fix the random state for the reproducibility.

    Returns
    -------
    data : array_like
        Data array of shape (n_epochs, n_sites, n_pts) array.
    roi : array_like
        Array of region of interest of shape (n_sites,)
    time : array_like
        The time vector of shape (n_times)
    """
    assert modality in ["meeg", "intra"]
    ma = np.array(CONFIG['MA_NAMES'])
    n_times += 100  # edge effect compensation
    # random state
    if not isinstance(random_state, int):
        random_state = np.random.randint(0, 100000)
    _rnd = np.random.RandomState(random_state)
    # number of sites / sources
    if modality == "intra":
        n_sites_per_roi = np.random.randint(1, n_sites_per_roi + 1)
    # select roi names
    select_roi = np.repeat(np.arange(n_roi), n_sites_per_roi)
    roi = ma[select_roi]
    n_sites = len(roi)
    # prepare variables
    signal = np.zeros((n_sites, n_epochs, n_times), dtype=float)
    time = np.arange(n_times).reshape(-1, 1) / sf
    f_sines = np.linspace(f_min, f_max, num=n_sines, endpoint=True)
    phy = _rnd.uniform(0., 2. * np.pi, (n_times, n_sines))
    sines = np.sin(2. * np.pi * f_sines.reshape(1, -1) * time + phy)
    amp_log = np.logspace(0, 1, n_sines, base=.1)
    # generate the data
    for k, i in product(range(n_sites), range(n_epochs)):
        amp = amp_log * _rnd.normal(0., 1., n_sines)
        sig = savgol_filter(np.dot(sines, amp), 21, 3)
        sig += _rnd.randn(*sig.shape) / (noise * sig.std())
        signal[k, i, :] = sig
    signal = np.moveaxis(zscore(signal, -1)[..., 50:-50], 0, 1)
    time = time[50:-50]
    if as_mne:
        from mne import create_info, EpochsArray
        info = create_info(roi.tolist(), sf, ch_types='seeg')
        signal = EpochsArray(signal, info, tmin=float(time[0]))
    return signal, roi, time.squeeze()


def sim_multi_suj_ephy(modality="meeg", n_subjects=10, **kwargs):
    """Simulate electrophysiological data of multiple subjects.

    Parameters
    ----------
    n_subjects : int | 10
        Number of subjects to simulate
    modality : {"meeg", "intra"}
        Recording modality. Can either be "meeg" (MEG / EEG) or "intra"
        (sEEG / ECoG)
    kwargs : dict | {}
        Additional arguments are send to the :sim_single_suj_ephy: function
    random_state : int | None
        Fix the random state for the reproducibility.

    Returns
    -------
    data : list
        List of length (n_subjects,) of data array each one having a shape of
        (n_epochs, n_sites, n_pts)
    roi : list
        List of length (n_subjects,) of arrays representing the region of
        interest's names per subject.
    time : array_like
        The time vector of shape (n_times)
    """
    # random state
    random_state = kwargs.get('random_state', None)
    if not isinstance(random_state, int):
        random_state = np.random.randint(0, 100000)
    # generate the data of all subjects
    data, roi = [], []
    for k in range(n_subjects):
        kwargs['random_state'] = random_state + k
        _data, _roi, time = sim_single_suj_ephy(modality=modality, **kwargs)
        data += [_data]
        roi += [_roi]
    return data, roi, time
