"""Wrapping functions."""
from frites import config
from frites.io import is_numba_installed

###############################################################################
# Numba wrapper

if config.CONFIG['USE_NUMBA'] and is_numba_installed():
    import numba
    def jit(signature=None, nopython=True, nogil=True, fastmath=True,  # noqa
            cache=True, **kwargs):
        return numba.jit(signature_or_function=signature, cache=cache,
                         nogil=nogil, fastmath=fastmath, nopython=nopython,
                         **kwargs)
else:
    def jit(*args, **kwargs):  # noqa
        def _jit(func):
            return func
        return _jit
