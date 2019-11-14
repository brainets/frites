"""Check if packages are installed."""


def is_pandas_installed(raise_error=False):
    """Test if pandas is installed."""
    try:
        import pandas  # noqa
        is_installed = True
    except ImportError:
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:
        raise IOError("pandas not installed. See https://pandas.pydata.org/#"
                      "best-way-to-install for installation instructions.")
    return is_installed


def is_xarray_installed(raise_error=False):
    """Test if xarray is installed."""
    try:
        import xarray  # noqa
        is_installed = True
    except ImportError:
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:
        raise IOError("xarray not installed. See http://xarray.pydata.org"
                      "for installation instructions.")
    return is_installed


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
