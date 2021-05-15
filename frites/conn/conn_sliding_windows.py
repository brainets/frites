"""Sliding windows for connectivity functions."""
from frites.io import set_log_level, logger

import numpy as np


def define_windows(times, windows=None, slwin_len=None, slwin_start=None,
                   slwin_stop=None, slwin_step=None, verbose=None):
    """Define temporal windows.

    This function can be used to either manually define temporal windows either
    automatic sliding windows. Note that every input parameters should be in
    the time domain (e.g seconds or milliseconds).

    Parameters
    ----------
    times : array_like
        Time vector
    windows : array_like | None
        Manual windows (e.g (.1, .2) or [(.1, .2), (.4, .5)]).
    slwin_len : float | None
        Length of each sliding (e.g .2 produces 200ms window length).
    slwin_start : float | None
        Time point for starting sliding windows (e.g 0.1). If None, sliding
        windows will start from the first time point.
    slwin_stop : float | None
        Time point for ending sliding windows (e.g 1.5). If None, sliding
        windows will finish at the last time point.
    slwin_step : float | None
        Temporal step between each temporal window (e.g .1 means that each
        consecutive windows are going to be separated by 100ms). This parameter
        can be used to define either overlapping or non-overlapping windows. If
        None, slwin_step is going to be set to slwin_step in order to produce
        consecutive non-overlapping windows.

    Returns
    -------
    win_sample : array_like
        Array of shape (n_windows, 2) of temporal indexes defining where each
        window (start, finish)
    mean_time : array_like
        Mean time vector inside each defined window of shape (n_windows,)

    See also
    --------
    plot_windows
    """
    set_log_level(verbose)
    assert isinstance(times, np.ndarray)
    logger.info("Defining temporal windows")
    stamp = times[1] - times[0]

    # -------------------------------------------------------------------------
    # build windows
    if (windows is None) and (slwin_len is None):
        logger.info("    No input detected. Full time window is used")
        win_time = np.array([[times[0], times[-1]]])
    elif windows is not None:
        logger.info("    Manual definition of windows")
        win_time = np.atleast_2d(windows)
    elif slwin_len is not None:
        # manage empty inputs
        if slwin_start is None: slwin_start = times[0]          # noqa
        if slwin_stop is None: slwin_stop = times[-1]           # noqa
        if slwin_step is None: slwin_step = slwin_len + stamp   # noqa
        logger.info(f"    Definition of sliding windows (len={slwin_len}, "
                    f"start={slwin_start}, stop={slwin_stop}, "
                    f"step={slwin_step})")
        # build the sliding windows
        sl_start = np.arange(slwin_start, slwin_stop - slwin_len, slwin_step)
        sl_stop = np.arange(slwin_start + slwin_len, slwin_stop, slwin_step)
        if len(sl_start) != len(sl_stop):
            min_len = min(len(sl_start), len(sl_stop))
            sl_start, sl_stop = sl_start[0:min_len], sl_stop[0:min_len]
        win_time = np.c_[sl_start, sl_stop]
    assert (win_time.ndim == 2) and (win_time.shape[1] == 2)

    # -------------------------------------------------------------------------
    # time to sample conversion
    win_sample = np.zeros_like(win_time, dtype=int)
    times = times.reshape(-1, 1)
    for n_k, k in enumerate(win_time):
        win_sample[n_k, :] = np.argmin(np.abs(times - k), axis=0)
    logger.info(f"    {win_sample.shape[0]} windows defined")

    return win_sample, win_time.mean(1)


def plot_windows(times, win_sample, x=None, title='', r_min=-.75, r_max=.75):
    """Simple plotting function for representing windows.

    Parameters
    ----------
    times : array_like
        Times vector of shape (n_times,)
    win_sample : array_like
        Windows in samples.
    x : array_like | None
        A signal to use as a background. If None, a pure sine is generated
        with 100ms period is generated
    title : string | ''
        String title to attach to the figure
    r_min, r_max : float | -.75, .75
        Window are represented by squares. Those two parameters can be used to
        control where box start and finish along the y-axis.

    Returns
    -------
    ax : gca
        The matplotlib current axes

    See also
    --------
    define_windows
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    if x is not None:
        # plot the sine
        plt.plot(times, x)
    # plot the windows
    win_time = times[win_sample]
    r_red, r_blue = [], []
    for n_k, k in enumerate(win_time):
        if n_k % 2 == 0:
            r_red += [Rectangle((k[0], r_min), k[1] - k[0], r_max - r_min)]
        elif n_k % 2 == 1:
            r_blue += [Rectangle((k[0], r_min), k[1] - k[0], r_max - r_min)]
    pc_blue = PatchCollection(r_blue, alpha=.5, color='blue')
    pc_red = PatchCollection(r_red, alpha=.5, color='red')
    plt.gca().add_collection(pc_blue)
    plt.gca().add_collection(pc_red)
    plt.title(title)
    plt.xlim(times[0], times[-1])

    return plt.gca()
