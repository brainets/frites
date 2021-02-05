"""Functions for managing input types of the DatasetEphy."""
import logging

import numpy as np

try:
    import neo
    HAS_NEO = True
except ModuleNotFoundError:
    HAS_NEO = False

from frites.io import set_log_level, logger
from frites.config import CONFIG


###############################################################################
###############################################################################
#                   MNE / XARRAY / NUMPY / NEO ---> NUMPY
###############################################################################
###############################################################################


def ds_ephy_io(x, roi=None, y=None, z=None, times=None, sub_roi=None,
               verbose=None):
    """Manage inputs conversion for the DatasetEphy.

    This function is used to convert NumPy / MNE / Xarray / Neo inputs into a
    standardized NumPy version.

    Parameters
    ----------
    x : list
        List of length (n_subjects,). Each element of the list should either be
        an array of shape (n_epochs, n_channels, n_times), mne.Epochs,
        mne.EpochsArray, mne.EpochsTFR (i.e. non-averaged power), DataArray or
        neo.Block. Note that each neo.Block has to contain (n_epochs) segments
        with one neo.AnalogSignal object each. AnalogSignals will be aligned by
        the first sample and only n_samples will be considered, where n_samples
        is the number of samples of the shortest AnalogSignal.
    roi : list | None
        List of length (n_subjects,) of roi names of length (n_channels)
    y, z : list | None
        List for the regressors. Each element should be an array of shape
        (n_epochs)
    sub_roi : list | None
        List of sub_roi names
    times : array_like | None
        Time vector

    Returns
    -------
    x : list
        List of data arrays each of shape (n_epochs, n_channels, n_times)
    y, z : list
        List of arrays each of shape (n_epochs,)
    roi : list
        List of arrays each of shape (n_channels,)
    times : array_like
        Time vector of shape (n_times,)
    sub_roi : array_like
        List of arrays each of shape (n_channels,)
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
        x, roi, y, z, times, sub_roi = xr_to_arr(
            x, roi=roi, y=y, z=z, times=times, sub_roi=sub_roi)
    elif 'neo.core' in str(type(x[0])):
        logger.info("    Neo inputs detected")
        if not HAS_NEO:
            raise ImportError('Can not convert neo ephy data.'
                              ' Neo can not be imported.')
        logger.info("    Converting neo inputs")
        x, times = neo_to_arr(x)
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
    if isinstance(sub_roi, list):
        assert all([k.shape == i.shape for k, i in zip(roi, sub_roi)])

    # -------------------------------------------------------------------------
    # categorical sub roi
    # -------------------------------------------------------------------------
    # concatenate everything and get unique elements
    if isinstance(sub_roi, list):
        import pandas as pd
        logger.info("    Replacing sub roi by categorical integers")

        # get unique sub roi and build replacement dict
        sub_roi_cat = np.r_[tuple([k.squeeze() for k in sub_roi])]
        sub_roi_u = np.unique(sub_roi_cat, return_index=True)
        repl = {k: v for k, v in zip(*sub_roi_u)}

        # replace for each subject
        sub_roi_int = []
        for _sub in sub_roi:
            sub_int = np.array(list(pd.Series(_sub).replace(repl, regex=True)))
            sub_roi_int += [sub_int]
    else:
        sub_roi_int = None

    return x, y, z, roi, times, sub_roi_int


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


def neo_to_arr(x):
    """Convert list of neo.Block objects into numpy arrays"""
    subject_list = []

    # identify number of epochs and channels
    n_epochs = min([len(bl.segments) for bl in x])
    n_times, *_, n_channels = x[0].segments[0].analogsignals[0].shape
    sampling_rate = x[0].segments[0].analogsignals[0].sampling_rate

    # consistency checks
    for bl in x:
        for seg in bl.segments:
            assert len(seg.analogsignals) == 1, \
            'Found more than 1 neo.AnalogSignal for a given Epoch.'
            assert n_channels == seg.analogsignals[0].shape[-1], \
            'AnalogSignals contain different numbers of channels.'
            assert sampling_rate == seg.analogsignals[0].sampling_rate,\
            'AnalogSignals have different sampling rates.'

            # find minimal number of samples
            n_times = min(n_times, len(seg.analogsignals[0]))

    # create 3D array for each subject
    for bl in x:
        subject_array = np.stack([seg.analogsignals[0].magnitude[:n_times] for seg in bl.segments])
        # reorder dimensions to match (n_epochs, n_channels, n_times)
        subject_list.append(subject_array.swapaxes(1, 2))

    times = x[0].segments[0].analogsignals[0].times[:n_times]
    return subject_list, times


def xr_to_arr(x, roi=None, y=None, z=None, times=None, sub_roi=None):
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
    if isinstance(sub_roi, str):
        coords['sub_roi'] = sub_roi
        sub_roi = [x[k].coords[sub_roi].data for k in range(len(x))]
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

    return x, roi, y, z, times, sub_roi


###############################################################################
###############################################################################
#                     MULTI TO UNIVARITE CONDITIONS
###############################################################################
###############################################################################

def multi_to_uni_cond_io(y):
    """Convert a vector of multi-conditions to a single one.

    Parameters
    ----------
    y : array_like
        Multi-variate array of discret values of shape (n_trials, n_conditions)

    Returns
    -------
    y : array_like
        Univariate array of discret values of shape (n_trials, n_conditions)
    """
    # evacuate single categories
    if y.ndim == 1 or (y.shape[1] == 1):
        return y
    n_trials, n_cond = y.shape

    # get unique multi-condition patterns
    idx = np.unique(y, axis=0, return_index=True)[1]
    u_cat = y[sorted(idx), :]

    # replace multi-conditions with a single integer
    y_uni = np.zeros((n_trials,), dtype=int)
    for n_p in range(u_cat.shape[0]):
        y_uni[np.all(y == u_cat[[n_p], ...], axis=1)] = n_p

    return y_uni



if __name__ == '__main__':
    y = np.array([[1, 0, 0, 0, 1, 1, 1, 2, 2],
                  ['a', 'a', 'a', 'b', 'b', 'b', 1, 1, 1]]).T
    print(y.shape)

    y_uni = multi_to_uni_cond_io(y)
    print(np.c_[y, y_uni])
