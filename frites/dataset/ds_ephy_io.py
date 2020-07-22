"""Functions for managing input types of the DatasetEphy."""
import logging

import numpy as np

from frites.io import set_log_level
from frites.config import CONFIG

logger = logging.getLogger("frites")


def ds_ephy_io(x, roi=None, y=None, z=None, times=None, verbose=None):
    """Manage inputs conversion for the DatasetEphy.

    This function is used to convert NumPy / MNE / Xarray inputs into a
    standardize NumPy version.

    Parameters
    ----------
    x : list
        List of length (n_subjects,). Each element of the list should either be
        an array of shape (n_epochs, n_channels, n_times), mne.Epochs,
        mne.EpochsArray, mne.EpochsTFR (i.e. non-averaged power) or DataArray
    roi : list | None
        List of length (n_subjects,) of roi names of length (n_channels)
    y, z : list | None
        List for the regressors. Each element should be an array of shape
        (n_epochs)
    times : array_like | None
        Time vector

    Returns
    -------
    x : list
        List of data array of shape (n_epochs, n_channels, n_times)
    y, z : list
        List of arrays of shape (n_epochs,)
    roi : list
        List of arrays of shape (n_channels,)
    times : array_like
        Time vector of shape (n_times,)
    """
    set_log_level(verbose)
    # -------------------------------------------------------------------------
    # data type detection and switch
    # -------------------------------------------------------------------------
    assert isinstance(x, list), ("x input should be a list of elements of "
                                 "length (n_subjects,)")
    assert all([type(x[k]) == type(x[0]) for k in range(len(x))]), (
        "All elements in the `x` inputs are not the same type")

    # -------------------------------------------------------------------------
    # conversion to array according to datatype
    # -------------------------------------------------------------------------
    if 'numpy' in str(type(x[0])):
        logger.info("    NumPy inputs detected")
    elif 'mne' in str(type(x[0])):
        logger.info("    Converting mne inputs")
        x, times, roi = mne_to_arr(x, roi=roi)
    elif 'xarray' in str(type(x[0])):
        logger.info("    Converting xarray inputs")
        x, roi, y, z, times = xr_to_arr(x, roi=roi, y=y, z=z, times=times)

    # -------------------------------------------------------------------------
    # manage none inputs
    # -------------------------------------------------------------------------
    # empty roi
    if not isinstance(roi, list):
        logger.warning("No roi have been provided. A default will be used "
                       "instead. You should use the `roi` input instead")
        roi = []
        for k in range(len(x)):
            roi += [np.array([f"roi_{i}" for i in range(x[k].shape[1])])]
    # empty time vector
    if not isinstance(times, np.ndarray):
        logger.warning("No time vector found. A default will be used instead."
                       " You should use the `times` input instead")
        times = np.arange(x[0].shape[-1])

    # -------------------------------------------------------------------------
    # shape and types checking before returning
    # -------------------------------------------------------------------------
    # spatio-temporal conversion
    roi = [np.asarray(roi[k]) for k in range(len(roi))]
    times = times.astype(np.float32)
    # data checking
    assert all([k.ndim in [3, 4] for k in x]), (
        "data should either contains 3d arrays (n_trials, n_channels, n_pts) "
        "or 4d arrays (n_trials, n_channels, n_freqs, n_pts)")
    x_sh = [x[k].shape for k in range(len(x))]
    is_sh_roi = [x_sh[k][1] == len(roi[k]) for k in range(len(x))]
    is_sh_times = [x_sh[k][-1] == len(times) for k in range(len(x))]
    assert all(is_sh_roi), "Inconsistent number of ROI"
    assert all(is_sh_times), "Inconsistent number of time points"
    assert all([list(x_sh[0])[1:] == list(x_sh[k])[1:]] for k in range(len(x)))
    if isinstance(y, list):
        y = [np.asarray(y[k]) for k in range(len(y))]
        assert len(y) == len(x), "length of y shoud be (n_subjects,)"
        assert [x_sh[k][0] == len(y[k]) for k in range(len(x))], (
            "Each element of the y input should have a length of (n_epochs,)")
    if isinstance(z, list):
        z = [np.asarray(z[k]) for k in range(len(z))]
        assert len(z) == len(x), "length of z shoud be (n_subjects,)"
        assert [x_sh[k][0] == len(z[k]) for k in range(len(x))], (
            "Each element of the z input should have a length of (n_epochs,)")

    return x, y, z, roi, times


def mne_to_arr(x, roi=None):
    """Convert list of MNE types into numpy arrays."""
    # get time vector and roi names (if empty)
    times = x[0].times
    if roi is None:
        logger.info("    Infer roi names using `ch_names`")
        roi = [np.asarray(x[k].ch_names) for k in range(len(x))]
    # get the data and replace inplace
    for k in range(len(x)):
        x[k] = x[k].get_data()

    return x, times, roi


def xr_to_arr(x, roi=None, y=None, z=None, times=None):
    """Xarray DataArray conversion to numpy arrays.

    The xarray supports using strings to specify the dimension name to use. Its
    also working for pandas MultiIndex.
    """
    coords = dict()
    if isinstance(times, str):
        coords['times'] = times
        times = eval(f"x[0].{times}.data")
    if isinstance(roi, str):
        coords['roi'] = roi
        roi = [x[k].coords[roi].data for k in range(len(x))]
    if isinstance(y, str):
        coords['y'] = y
        y = [x[k].coords[y].data for k in range(len(x))]
    if isinstance(z, str):
        coords['z'] = z
        z = [x[k].coords[z].data for k in range(len(x))]
    if coords:
        log_str = f"\n{' ' * 8}".join([f"{k}: {v}" for k, v in coords.items()])
        logger.info(f"    The following coordinates have been used : \n"
                    f"{' ' * 8}{log_str}")
    x = [x[k].data for k in range(len(x))]

    return x, roi, y, z, times
