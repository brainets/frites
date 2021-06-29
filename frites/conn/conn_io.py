"""Deal with connectivity inputs."""
import numpy as np
import pandas as pd
import xarray as xr
import mne

from frites.io import set_log_level, logger
from frites.config import CONFIG
from frites.dataset import SubjectEphy


def conn_io(data, times=None, roi=None, y=None, sfreq=None, agg_ch=False,
            win_sample=None, pairs=None, block_size=None, name=None,
            sort=True, verbose=None):
    """Prepare connectivity variables.

    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported :

            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)

    times : array_like | None
        Time vector array of shape (n_times,). If the input is an xarray, the
        name of the time dimension can be provided
    roi : array_like | None
        ROI names of a single subject. If the input is an xarray, the
        name of the ROI dimension can be provided
    y : array_like | None
        A variable to attach to the trials
    sfreq : float | None
        Sampling frequency
    win_sample : array_like | None
        Array of shape (n_windows, 2) describing where each window start and
        finish. You can use the function :func:`frites.conn.define_windows`
        to define either manually either sliding windows. If None, the entire
        time window is used instead.
    pairs : array_like | None
        Pairs of contacts
    agg_ch : bool | False
        In case there are multiple electrodes, channels, contacts or sources
        inside a brain region, specify how the data has to be aggregated.
    block_size : int | None
        Number of blocks of trials to process at once.
    sort : bool | True
        Specify whether roi names should be sorted (True for undirected
        measures) or not (False for directed measures)

    Returns
    -------
    data : xarrayr.DataArray
        Data converted to DataArray
    cfg : dict
        Additional parameters
    """
    set_log_level(verbose)
    logger.debug(f"Prepare inputs for computing {name}")
    cfg = dict()

    # ____________________________ DATA CONVERSION ____________________________
    # keep xarray attributes and trials
    if isinstance(data, xr.DataArray):
        trials, attrs = data[data.dims[0]].data, data.attrs
    else:
        if isinstance(data, (mne.EpochsArray, mne.Epochs)):
            n_trials = data._data.shape[0]
        else:
            n_trials = data.shape[0]
        trials, attrs = np.arange(n_trials), {}
    if y is None:
        y = trials

    # main data conversion
    data = SubjectEphy(data, y=y, roi=roi, times=times, sfreq=sfreq,
                       verbose=verbose)
    roi, times = data['roi'].data, data['times'].data
    trials = data['y'].data
    n_trials = len(trials)

    cfg['sfreq'], cfg['attrs'] = data.attrs['sfreq'], attrs

    # _________________________________ SPACE _________________________________
    # get indices of pairs of (group) regions
    if (pairs is None):
        if agg_ch:  # for multi-variate computations
            gp = pd.DataFrame({'roi': roi}).groupby('roi', sort=False).groups
            roi_gp = np.array(list(gp.keys()))
            roi_idx = [list(k) for k in gp.values()]
        else:
            roi_gp, roi_idx = roi, np.arange(len(roi)).reshape(-1, 1)
        x_s, x_t = np.triu_indices(len(roi_gp), k=1)
    else:
        assert isinstance(pairs, np.ndarray)
        assert (pairs.ndim == 2) and (pairs.shape[1] == 2)
        x_s, x_t = pairs[:, 0], pairs[:, 1]
        roi_gp, roi_idx = roi, np.arange(len(roi)).reshape(-1, 1)

    # build names of pairs of brain regions (case insensitive)
    roi_c = np.c_[roi_gp[x_s], roi_gp[x_t]]
    if sort:
        idx = np.argsort(np.char.lower(roi_c.astype(str)), axis=1)
        roi_s, roi_t = np.c_[[r[i] for r, i in zip(roi_c, idx)]].T
    else:
        roi_s, roi_t = roi_c.T
    roi_p = [f"{s}-{t}" for s, t in zip(roi_s, roi_t)]

    logger.debug(f"    Spatial dimension (n_groups={len(roi_gp)}; "
                 f"n_pairs={len(roi_p)})")
    cfg['roi_p'], cfg['roi_idx'] = roi_p, roi_idx
    cfg['x_s'], cfg['x_t'] = x_s, x_t

    # _________________________________ TIME __________________________________
    if win_sample is None:
        win_sample = np.array([[0, len(times) - 1]])
    assert isinstance(win_sample, np.ndarray) and (win_sample.ndim == 2)
    assert win_sample.dtype in CONFIG['INT_DTYPE']

    logger.debug(f"    Time dimension (n_windows={win_sample.shape[0]})")
    cfg['win_sample'] = win_sample
    cfg['win_times'] = times[win_sample].mean(1)

    # ______________________________ FREQUENCY ________________________________

    # ______________________________ BLOCK-SIZE _______________________________
    # build block size indices
    if isinstance(block_size, int) and (block_size > 1):
        blocks = np.array_split(np.arange(n_trials), block_size)
    else:
        blocks = [np.arange(n_trials)]

    logger.debug(f"    Block of trials (n_blocks={len(blocks)})")
    cfg['blocks'] = blocks

    return data, cfg
