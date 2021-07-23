"""Pre- and post-processing functions."""
import numpy as np
import xarray as xr
from scipy.signal import savgol_filter as savgol
from scipy.signal import fftconvolve

from frites.io import set_log_level, logger


def savgol_filter(x, h_freq, axis=None, sfreq=None, polyorder=5, verbose=None):
    """Filter the data using Savitzky-Golay polynomial method.

    This function is an adaptation of the mne-python one for xarray.DataArray.

    Parameters
    ----------
    x : array_like
        Multidimensional array or DataArray
    h_freq : float
        Approximate high cut-off frequency in Hz. Note that this is not an
        exact cutoff, since Savitzky-Golay filtering is done using
        polynomial fits instead of FIR/IIR filtering. This parameter is
        thus used to determine the length of the window
    axis : int, string | None
        Position of the time axis. Can either be an integer when `x` is a
        NumPy array or a string (e.g 'times') when using a DataArray
    polyorder : int | 5
        Polynomial order

    Returns
    -------
    x_filt : array_like
        Filtered data

    Notes
    -----
    For Savitzky-Golay low-pass approximation, see:
        https://gist.github.com/larsoner/bbac101d50176611136b

    See also
    --------
    kernel_smoothing
    """
    set_log_level(verbose)
    # inputs checking
    if isinstance(x, xr.DataArray):
        dims = list(x.dims)
        # get axis name
        if axis is None:
            axis = 'times'
        if isinstance(axis, str):
            axis = list(x.dims).index(axis)
        # get sfreq if possible
        if not isinstance(sfreq, (int, float)):
            assert 'times' in dims
            sfreq = 1. / (x['times'].data[1] - x['times'].data[0])
    assert isinstance(h_freq, (int, float))
    assert isinstance(axis, int)
    assert isinstance(sfreq, (int, float))
    if h_freq >= sfreq / 2.:
        raise ValueError('h_freq must be less than half the sample rate')

    # get window length
    window_length = (int(np.round(sfreq / h_freq)) // 2) * 2 + 1
    logger.info(f'    Using savgol length {window_length}')

    # apply savgol depending on input type
    kw = dict(axis=axis, polyorder=polyorder, window_length=window_length)
    if isinstance(x, xr.DataArray):
        x.data = savgol(x.data, **kw)
        return x
    else:
        return savgol(x, **kw)


def kernel_smoothing(x, kernel, axis=-1):
    """Apply a kernel smoothing using the fftconvolve.

    Parameters
    ----------
    x : array_like
        Array to smooth. It can also be an instance of xarray.DataArray.
    kernel : array_like
        Kernel to use for smoothing (e.g kernel=np.ones((10,)) for a moving
        average or np.hanning(10))
    axis : int, string | -1
        The axis to use for smoothing (typically the time-axis). If the x input
        is a DataArray, axis can be a string (e.g axis='times')

    Returns
    -------
    x : array_like
        The smoothed array with the same shape and type as the input x.

    See also
    --------
    savgol_filter
    """
    assert isinstance(kernel, np.ndarray) and (kernel.ndim == 1)
    # get axis number when dataarray
    if isinstance(x, xr.DataArray) and isinstance(axis, str):
        axis = x.get_axis_num(axis)
    axis = np.arange(x.ndim)[axis]
    # reshape kernel to match input number of dims
    k_sh = [1 if k != axis else len(kernel) for k in range(x.ndim)]
    kernel = kernel.reshape(*tuple(k_sh))
    # apply the convolution
    kw = dict(axes=axis, mode='same')
    if isinstance(x, xr.DataArray):
        x.data = fftconvolve(x.data, kernel, **kw)
    else:
        x = fftconvolve(x, kernel, **kw)
    return x


def nonsorted_unique(data, assert_unique=False):
    """Get an unsorted version of a element.

    Parameters
    ----------
    data : array_like
        Array of data, list etc.
    assert_unique : bool | False
        Raise an error if the input doesn't contain unique elements

    Returns
    -------
    u_data : array_like
        Array of non-sorted unique elements
    """
    data = np.asarray(data)
    if assert_unique:
        assert len(np.unique(data)) == len(data), (
            "Brain regions are not unique for inferring connectivity pairs")
    _, u_idx = np.unique(data, return_index=True)
    return data[np.sort(u_idx)]


def time_to_sample(values, sf=None, times=None, round='closer', verbose=None):
    """Convert values from time to sample space.

    Parameters
    ----------
    values : array_like
        Array of values to convert.
    sf : float | None
        Sampling frequency. If None, the time vector can also be supplied to
        infer it.
    times : array_like | None
        Time vector used to infer the sampling frequency.
    round : {'lower', 'closer', 'upper'}
        Use either lower, closer or upper rounding. Default is closer

    Returns
    -------
    values : array_like
        Values converted to sample space.
    """
    set_log_level(verbose)
    # get sampling frequency
    if isinstance(times, (list, tuple, np.ndarray)):
        sf = 1. / (times[1] - times[0])
        logger.info(f"Inferring sampling rate : {sf}")
    assert isinstance(sf, (int, float)), "Sampling rate is missing"

    # array conversion
    values = np.asarray(values)
    fcn = {'lower': np.floor, 'closer': np.round, 'upper': np.ceil}[round]
    values_i = fcn(values * sf).astype(int)

    return values_i


def get_closest_sample(ref, values, precision=None, return_precision=False):
    """Get the sample of closest value in a reference time vector.

    Parameters
    ----------
    ref : array_like
        Reference vector
    values : array_like
        Values to seek in the reference vector
    precision : float | None
        Minimum precision to achieve.
    return_precision : {True, False}
        If true, the precision of length (n_values,) is also returned

    Returns
    -------
    sample : array_like
        Array of length (n_values,) containing the sample of closest values in
        the reference vector
    precisions : array_like
        If return_precision, the vector of precisions of length (n_values,) is
        also returned
    """
    # array conversion
    ref, values = np.asarray(ref), np.asarray(values)

    # find closest sample
    diff = np.abs(ref.reshape(-1, 1) - values.reshape(1, -1)).argmin(0)
    precisions = np.abs(values - ref[diff])

    if isinstance(precision, (int, float)):

        assert precision > 0, "Precision should be strictly positive"
        assert np.all(precisions < precision), "Precision not sufficient"

    if return_precision:
        return (diff, precisions)
    else:
        return diff
