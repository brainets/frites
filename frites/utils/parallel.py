"""Parallel function wrapper."""
from joblib import Parallel, delayed

from mne.utils import ProgressBar


def parallel_func(fcn, n_jobs=-1, verbose=None, total=None, **kwargs):
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

    # parallel functions
    para_fcn = delayed(fcn)
    parallel = Parallel(n_jobs, **kwargs)

    if total is not None:
        def parallel_progress(op_iter):
            return parallel(ProgressBar(iterable=op_iter, max_value=total))
        parallel_out = parallel_progress
    else:
        parallel_out = parallel

    return parallel_out, para_fcn
