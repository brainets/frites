"""Utility functions for measuring performances."""
import numpy as np
from time import time as tst


def timeit(method):
    """Get computing time of a function.

    @timeit
    def my_fcn()
    """
    def timed(*args, **kw):
        ts = tst()
        result = method(*args, **kw)
        te = tst()
        print(f"{method.__name__} : {(te - ts) * 1000}ms")
        return result
    return timed


def id(x):
    """Get the memory block address of an array."""
    return x.__array_interface__['data'][0]


def get_data_base(arr):
    """For a given array, finds the base array that "owns" the actual data."""
    base = arr
    while isinstance(base.base, np.ndarray):
        base = base.base
    return base


def arrays_share_data(x, y):
    """Return if two arrays share an offset."""
    return get_data_base(x) is get_data_base(y)
