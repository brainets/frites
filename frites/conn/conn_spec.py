"""Compute single-trial spectral connectivity.

Authors : Vinicius Lima <vinicius.lima.cordeiro@gmail.com >
          Etienne Combrisson <e.combrisson@gmail.com>

License : BSD (3-clause)
"""
import numpy as np
import xarray as xr
from mne_connectivity import spectral_connectivity_time

from frites.conn import conn_io
from frites.io import set_log_level, logger, check_attrs
from frites.utils import parallel_func
from frites.conn.conn_tf import (_tf_decomp, _create_kernel,
                                 _smooth_spectra, _foi_average)


###############################################################################
###############################################################################
#                               CORE FUNCTIONS
###############################################################################
###############################################################################

def _coh(w, kernel, foi_idx, x_s, x_t, kw_para):
    """Pairwise coherence."""
    # auto spectra (faster that w * w.conj())
    s_auto = w.real ** 2 + w.imag ** 2

    # smooth the auto spectra
    s_auto = _smooth_spectra(s_auto, kernel)

    # define the pairwise coherence
    def pairwise_coh(w_x, w_y):
        # computes the coherence
        s_xy = w[:, w_y, :, :] * np.conj(w[:, w_x, :, :])
        s_xy = _smooth_spectra(s_xy, kernel)
        s_xx = s_auto[:, w_x, :, :]
        s_yy = s_auto[:, w_y, :, :]
        out = np.abs(s_xy) ** 2 / (s_xx * s_yy)
        # mean inside frequency sliding window (if needed)
        if isinstance(foi_idx, np.ndarray):
            return _foi_average(out, foi_idx)
        else:
            return out

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(pairwise_coh, **kw_para)

    # compute the single trial coherence
    return parallel(p_fun(s, t) for s, t in zip(x_s, x_t))


def _plv(w, kernel, foi_idx, x_s, x_t, kw_para):
    """Pairwise phase-locking value."""
    # define the pairwise plv
    def pairwise_plv(w_x, w_y):
        # computes the plv
        s_xy = w[:, w_y, :, :] * np.conj(w[:, w_x, :, :])
        # complex exponential of phase differences
        exp_dphi = s_xy / np.abs(s_xy)
        # smooth e^(-i*\delta\phi)
        exp_dphi = _smooth_spectra(exp_dphi, kernel)
        # computes plv
        out = np.abs(exp_dphi)
        # mean inside frequency sliding window (if needed)
        if isinstance(foi_idx, np.ndarray):
            return _foi_average(out, foi_idx)
        else:
            return out

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(pairwise_plv, **kw_para)

    # compute the single trial coherence
    return parallel(p_fun(s, t) for s, t in zip(x_s, x_t))


def _cs(w, kernel, foi_idx, x_s, x_t, kw_para):
    """Pairwise cross-spectra."""
    # define the pairwise cross-spectra
    def pairwise_cs(w_x, w_y):
        #  computes the cross-spectra
        out = w[:, w_x, :, :] * np.conj(w[:, w_y, :, :])
        out = _smooth_spectra(out, kernel)
        if foi_idx is not None:
            return _foi_average(out, foi_idx)
        else:
            return out

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(pairwise_cs, **kw_para)

    # compute the single trial coherence
    return parallel(p_fun(s, t) for s, t in zip(x_s, x_t))


###############################################################################
###############################################################################
#                               MAIN FUNCTION
###############################################################################
###############################################################################


def conn_spec(
        data, freqs=None, metric='coh', roi=None, times=None, pairs=None,
        sfreq=None, foi=None, sm_times=.5, sm_freqs=1, sm_kernel='hanning',
        mode='morlet', n_cycles=7., mt_bandwidth=None, decim=1, kw_cwt={},
        kw_mt={}, block_size=None, n_jobs=-1, verbose=None, dtype=np.float32):
    """Wavelet-based single-trial time-resolved spectral connectivity.

    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported :

            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)

    metric : str | "coh"
        Which connectivity metric. Use either :

            * 'coh' : Coherence
            * 'plv' : Phase-Locking Value (PLV)
            * 'sxy' : Cross-spectrum

        By default, the coherenc is used.
    freqs : array_like
        Array of central frequencies of shape (n_freqs,).
    roi : array_like | None
        ROI names of a single subject. If the input is an xarray, the
        name of the ROI dimension can be provided
    times : array_like | None
        Time vector array of shape (n_times,). If the input is an xarray, the
        name of the time dimension can be provided
    pairs : array_like | None
        Pairs of links of shape (n_pairs, 2) to compute. If None, all pairs are
        computed
    sfreq : float | None
        Sampling frequency
    foi : array_like | None
        Extract frequencies of interest. This parameters should be an array of
        shapes (n_foi, 2) defining where each band of interest start and
        finish.
    sm_times : float | .5
        Number of points to consider for the temporal smoothing in seconds. By
        default, a 500ms smoothing is used.
    sm_freqs : int | 1
        Number of points for frequency smoothing. By default, 1 is used which
        is equivalent to no smoothing
    kernel : {'square', 'hanning'}
        Kernel type to use. Choose either 'square' or 'hanning'
    mode : {'morlet', 'multitaper'}
        Spectrum estimation mode can be either: 'multitaper' or 'morlet'.
    n_cycles : array_like | 7.
        Number of cycles to use for each frequency. If a float or an integer is
        used, the same number of cycles is going to be used for all frequencies
    mt_bandwidth : array_like | None
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
    block_size : int | None
        Number of blocks of trials to process at once. This parameter can be
        use in order to decrease memory load. If None, all trials are used. If
        for example block_size=2, the number of trials are subdivided into two
        groups and each group is process one after the other.
    n_jobs : int | 1
        Number of jobs to use for parallel computing (use -1 to use all
        jobs). The parallel loop is set at the pair level.

    Returns
    -------
    conn : xarray.DataArray
        DataArray of shape (n_trials, n_pairs, n_freqs, n_times)
    """
    set_log_level(verbose)

    if isinstance(sm_times, np.ndarray):
        raise NotImplementedError("Frequency dependent kernel in development"
                                  f"only first {sm_times[0]} will be used")

    if mode == 'morlet':
        mode = 'cwt_morlet'
    conn = spectral_connectivity_time(
        data=data, names=roi, method=metric, sfreq=sfreq,
        foi=foi, sm_times=sm_times, sm_freqs=sm_freqs, sm_kernel=sm_kernel,
        mode=mode, n_cycles=n_cycles, mt_bandwidth=mt_bandwidth, freqs=freqs,
        block_size=block_size, n_jobs=n_jobs, verbose=verbose)

    # _________________________________ INPUTS ________________________________
    # inputs conversion
    data, cfg = conn_io(
        data, times=times, roi=roi, agg_ch=False, win_sample=None, pairs=pairs,
        sort=True, block_size=block_size, sfreq=sfreq, freqs=freqs, foi=foi,
        sm_times=sm_times, sm_freqs=sm_freqs, verbose=verbose,
    )

    # extract variables
    _, trials, attrs = data.data, data['y'].data, cfg['attrs']
    times, _ = data['times'].data, len(trials)
    _, _, roi_p = cfg['x_s'], cfg['x_t'], cfg['roi_p']
    _, sfreq = cfg['blocks'], cfg['sfreq']
    freqs, _, _ = cfg['freqs'], cfg['need_foi'], cfg['foi_idx']
    f_vec, sm_times, sm_freqs = cfg['f_vec'], cfg['sm_times'], cfg['sm_freqs']

    # temporal decimation
    if isinstance(decim, int):
        times = times[::decim]
        sm_times = int(np.round(sm_times / decim))
        sm_times = max(sm_times, 1)

    # _________________________________ OUTPUTS _______________________________
    # configuration
    cfg = dict(
        sfreq=sfreq, sm_times=sm_times, sm_freqs=sm_freqs, sm_kernel=sm_kernel,
        mode=mode, n_cycles=n_cycles, mt_bandwidth=mt_bandwidth, decim=decim,
        type=metric
    )

    # conversion
    conn = xr.DataArray(conn.get_data(), dims=('trials', 'roi', 'freqs', 'times'),
                        name=metric, coords=(trials, roi_p, f_vec, times),
                        attrs=check_attrs({**attrs, **cfg}))
    return conn


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_trials = 100
    n_roi = 3
    n_times = 1000
    sfreq = 128.
    nt = int(np.round(n_trials / 2))

    # trials = np.random.randint(0, 1, (n_trials,))
    trials = [0] * nt + [1] * nt
    roi = [f"r{k}" for k in range(n_roi)]
    times = np.arange(n_times) / sfreq
    s1, s2 = slice(0, 500), slice(500, 1000)

    """
    - 25hz coherence between samples [0, 500]
    - 40hz coherence between samples [500, 1000]
    """
    x = np.random.rand(n_trials, n_roi, n_times)
    x[:nt, ..., s1] += np.sin(2 * np.pi * 25 * times[s1]).reshape(1, 1, -1)
    x[nt::, ..., s2] += np.sin(2 * np.pi * 40 * times[s2]).reshape(1, 1, -1)

    x = xr.DataArray(x, dims=('trials', 'roi', 'times'),
                     coords=(trials, roi, times))
    freqs = np.linspace(2, 60, 100)
    n_cycles = freqs / 2.

    foi = np.array([[2, 4], [5, 7], [8, 13], [13, 30], [30, 60]])
    coh = conn_spec(
        x, freqs, sfreq=sfreq, roi='roi', times='times', sm_times=2.,
        sm_freqs=1, mode='morlet', n_cycles=n_cycles,
        decim=1, foi=None, block_size=4, n_jobs=1, metric='plv')

    coh.groupby('trials').mean('trials').plot.imshow(
        x='times', y='freqs', col='roi', row='trials')
    plt.show()
