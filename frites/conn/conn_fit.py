"""Feature specific information transfer (Numba compliant)."""
import numpy as np

from frites.utils import jit


@jit("f4[:,:,:](f4[:,:,:], f4[:,:,:], f4[:], f4)")
def conn_fit(x_s, x_t, times, max_delay):  # noqa
    """Compute Feature-specific Information Transfer (FIT).

    This function has been written for supporting 3D arrays. If Numba is
    installed, performances of this function can be greatly improved.

    Parameters
    ----------
    x_s : array_like
        Array to use as source. Must be a 3d array of shape (:, :, n_times)
        and of type np.float32
    x_t : array_like
        Array to use as target. Must be a 3d array of shape (:, :, n_times)
        and of type np.float32
    times : array_like
        Time vector of shape (n_times,) and of type np.float32
    max_delay : float | .3
        Maximum delay (must be a np.float32)

    Returns
    -------
    fit : array_like
        Array of FIT of shape (:, :, n_times - max_delay)
    """
    # ---------------------------------------------------------------------
    n_dim, n_suj, n_times = x_s.shape
    # time indices for target roi
    t_start = np.where(times > times[0] + max_delay)[0]
    # max delay index
    max_delay = n_times - len(t_start)

    # ---------------------------------------------------------------------
    # Compute FIT on original MI values
    fit = np.zeros((n_dim, n_suj, n_times - max_delay), dtype=np.float32)

    # mi at target roi in the present
    x_t_pres = x_t[:, :, t_start]

    # Loop over delays for past of target and sources
    for delay in range(1, max_delay):
        # get past delay indices
        past_delay = t_start - delay
        # mi at target roi in the past
        x_t_past = x_t[:, :, past_delay]
        # mi at sources roi in the past
        x_s_past = x_s[:, :, past_delay]
        # redundancy between sources and target (min MI)
        red_s_t = np.minimum(x_t_pres, x_s_past)
        # redundancy between sources, target present and target past
        red_all = np.minimum(red_s_t, x_t_past)
        # sum delay-specific FIT (source, target)
        fit += red_s_t - red_all

    return fit
