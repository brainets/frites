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
