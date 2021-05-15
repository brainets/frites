"""Parallel function wrapper."""
import os

from joblib import Parallel, delayed, Memory

from mne.utils import ProgressBar

from frites.io import logger


def parallel_func(fcn, n_jobs=-1, verbose=None, total=None, mesg=None,
                  cache_dir=None, **kwargs):
    """Get an instance of parallel and delayed function.

    This function is inspired by MNE's one.

    Parameters
    ----------
    func : callable
        A function.
    n_jobs : int
        Number of jobs to run in parallel.
    total : int | None
        If int, use a progress bar to display the progress of dispatched
        jobs. This should only be used when directly iterating, not when
        using ``split_list`` or :func:`np.array_split`.
        If None (default), do not add a progress bar.
    mesg : string | None
        Message to display on the progress bar
    cache_dir : string | None
        If path to an existing directory, the function is going to cache the
        computations
    kwargs : dict | {}
        Additional arguments are sent to the joblibe.Parallel function.

    Returns
    -------
    parallel: instance of joblib.Parallel or list
        The parallel object.
    my_func: callable
        ``func`` if not parallel or delayed(func).
    """
    from frites.config import CONFIG

    # manually merge inputs inside the default config
    for k, v in CONFIG["JOBLIB_CFG"].copy().items():
        kwargs[k] = v
    # verbosity level of joblib
    kwargs['verbose'] = 1 if verbose in ['debug', True] else 0

    # caching option
    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        logger.info(f'Caching computations to {cache_dir}')
        memory = Memory(cache_dir, verbose=kwargs['verbose'])
        fcn = memory.cache(fcn)

    # parallel functions
    para_fcn = delayed(fcn)
    parallel = Parallel(n_jobs=n_jobs, **kwargs)

    if total is not None:
        def parallel_progress(op_iter):
            return parallel(ProgressBar(iterable=op_iter, max_value=total,
                                        mesg=mesg))
        parallel_out = parallel_progress
    else:
        parallel_out = parallel

    return parallel_out, para_fcn
