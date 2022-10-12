"""Compute time-frequency decomposition base on Morlet or Multitaper methods.

This script contains the function:

1. _tf_decomp used to decompose the sinal in tf domains using Morlet or
    Multitaper
1. _create_kernel: Create a kernel to smooth the spectra (either boxcar or
    hanning)
2. _smooth_kernel: Perform the smoothing operation on the spectra based on the
   convolution theorem
"""
# Authors : Vinicius Lima <vinicius.lima.cordeiro@gmail.com >
#           Etienne Combrisson <e.combrisson@gmail.com>
#
# License : BSD (3-clause)


import numpy as np

from mne.time_frequency import tfr_array_morlet, tfr_array_multitaper
from scipy.signal import fftconvolve


def _tf_decomp(data, sf, freqs, mode='morlet', n_cycles=7.0, mt_bandwidth=None,
               decim=1, kw_cwt={}, kw_mt={}, n_jobs=1):
    """Time-frequency decomposition using MNE-Python.

    Parameters
    ----------
    data : array_like
        Electrophysiological data of shape (n_trials, n_chans, n_times)
    sf : float
        Sampling frequency
    freqs : array_like
        Central frequency vector.
    mode : {'morlet', 'multitaper'}
        Spectrum estimation mode can be either: 'multitaper' or 'morlet'.
    n_cycles : array_like | 7.
        Number of cycles to use for each frequency. If a float or an integer is
        used, the same number of cycles is going to be used for all frequencies
    mt_bandwidth : int | float | array_like | None
        The bandwidth of the multitaper windowing function in Hz. Only used in
        'multitaper' mode.
    decim : int | 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition. default 1 If int, returns tfr[…, ::decim]. If slice,
        returns tfr[…, decim].
    kw_cwt : dict | {}
        Additional arguments sent to the mne-function
        :py:`mne.time_frequency.tfr_array_morlet`
    kw_mt : dict | {}
        Additional arguments sent to the mne-function
        :py:`mne.time_frequency.tfr_array_multitaper`

    Returns
    -------
    out : array_like
        Time-frequency transform of shape (n_epochs, n_chans, n_freqs, n_times)
    """
    if mode == 'morlet':
        out = tfr_array_morlet(
            data, sf, freqs, n_cycles=n_cycles, output='complex', decim=decim,
            n_jobs=n_jobs, **kw_cwt)
    elif mode == 'multitaper':
        # In case multiple values are provided for mt_bandwidth
        # the MT decomposition is done separatedly for each
        # Frequency center
        if isinstance(mt_bandwidth, (list, tuple, np.ndarray)):
            # Arrays freqs, n_cycles, mt_bandwidth should have the same size
            assert len(freqs) == len(n_cycles) == len(mt_bandwidth)
            out = []
            for f_c, n_c, mt in zip(freqs, n_cycles, mt_bandwidth):
                _out = tfr_array_multitaper(
                    data, sf, [f_c], n_cycles=float(n_c), time_bandwidth=mt,
                    output='complex', decim=decim, n_jobs=n_jobs, **kw_mt
                )

                # recent version of mne allows to return the TF decomposition
                # with the additional n_tapers dimension. This patch takes the
                # mean over the tapers
                if _out.ndim == 5:
                    _out = _out.mean(2)

                out.append(_out)

            # stack everything
            out = np.stack(out, axis=2).squeeze()
        elif isinstance(mt_bandwidth, (type(None), int, float)):
            out = tfr_array_multitaper(
                data, sf, freqs, n_cycles=n_cycles,
                time_bandwidth=mt_bandwidth, output='complex', decim=decim,
                n_jobs=n_jobs, **kw_mt)

            # recent version of mne allows to return the TF decomposition
            # with the additional n_tapers dimension. This patch takes the
            # mean over the tapers
            if out.ndim == 5:
                out = out.mean(2)

    else:
        raise ValueError('Method should be either "morlet" or "multitaper"')

    return out


###############################################################################
###############################################################################
#                         SPECTRA SMOOTHING METHODS
###############################################################################
###############################################################################


def _create_kernel(sm_times, sm_freqs, kernel='hanning'):
    """2D (freqs, time) smoothing kernel.

    Parameters
    ----------
    sm_times : int, array_like
        Number of points to consider for the temporal smoothing,
        if it is an array it will be considered that the kernel
        if frequence dependent.
    sm_freqs : int
        Number of points to consider for the frequency smoothing
    kernel : {'square', 'hanning'}
        Kernel type to use. Choose either 'square' or 'hanning'

    Returns
    -------
    kernel : array_like
        Smoothing kernel of shape (sm_freqs, sm_times)
    """
    # frequency dependent kernels
    if isinstance(sm_times, (np.ndarray, list, tuple)):
        sm_freqs = 1  # force 1hz smoothing
        kernels = [_create_kernel(
            sm, sm_freqs, kernel=kernel) for sm in sm_times]
        return kernels

    # frequency independent kernels
    if kernel == 'square':
        return np.full((sm_freqs, sm_times), 1. / (sm_times * sm_freqs))
    elif kernel == 'hanning':
        hann_t, hann_f = np.hanning(sm_times), np.hanning(sm_freqs)
        hann = hann_f.reshape(-1, 1) * hann_t.reshape(1, -1)
        return hann / np.sum(hann)
    else:
        raise ValueError(f"No kernel {kernel}")


def _smooth_spectra(spectra, kernel, scale=False, decim=1):
    """Smoothing spectra.

    This function assumes that the frequency and time axis are respectively
    located at positions (..., freqs, times).

    Parameters
    ----------
    spectra : array_like
        Spectra of shape (..., n_freqs, n_times)
    kernel : array_like, list
        Smoothing kernel  (or list of kernels) of shape (sm_freqs, sm_times)
    decim : int | 1
        Decimation factor to apply after the kernel smoothing

    Returns
    -------
    sm_spectra : array_like
        Smoothed spectra of shape (..., n_freqs, n_times)
    """
    # define axes to use for smoothing
    axes = -1 if scale else (-2, -1)

    # frequency (in)dependent smoothing
    if isinstance(kernel, list):
        for n_k, kern in enumerate(kernel):
            spectra[..., n_k, :] = __smooth_spectra(
                spectra[..., n_k, :], kern, axes)
    else:
        spectra = __smooth_spectra(spectra, kernel, axes)

    # return decimated spectra
    return spectra[..., ::decim]


def __smooth_spectra(spectra, kernel, axes):
    """Single kernel smoothing."""
    # fill potentially missing dimensions
    while kernel.ndim != spectra.ndim:
        kernel = kernel[np.newaxis, ...]

    # smooth the spectra
    return fftconvolve(spectra, kernel, mode='same', axes=axes)


def _foi_average(conn, foi_idx):
    """Average inside frequency bands.

    The frequency dimension should be located at -2.

    Parameters
    ----------
    conn : np.ndarray
        Array of shape (..., n_freqs, n_times)
    foi_idx : array_like
        Array of indices describing frequency bounds of shape (n_foi, 2)

    Returns
    -------
    conn_f : np.ndarray
        Array of shape (..., n_foi, n_times)
    """
    # get the number of foi
    n_foi = foi_idx.shape[0]

    # get input shape and replace n_freqs with the number of foi
    sh = list(conn.shape)
    sh[-2] = n_foi

    # compute average
    conn_f = np.zeros(sh, dtype=conn.dtype)
    for n_f, (f_s, f_e) in enumerate(foi_idx):
        conn_f[..., n_f, :] = conn[..., f_s:f_e, :].mean(-2)
    return conn_f
