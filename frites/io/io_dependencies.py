"""Check if packages are installed."""


def is_numba_installed(raise_error=False):
    """Test if numba is installed."""
    try:
        import numba  # noqa
        is_installed = True
    except ImportError:
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:
        raise IOError("numba not installed. See http://numba.pydata.org/ for "
                      "installation instructions.")
    return is_installed
