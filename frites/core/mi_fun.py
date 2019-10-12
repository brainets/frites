"""MI functions for ffx / rfx inferences."""
import numpy as np

from frites.config import CONFIG
from frites.core import mi_nd_gg


###############################################################################
#                        I(CONTINUOUS; CONTINUOUS)
###############################################################################


def mi_gg_ffx(x, y, z, suj):
    """I(C; C) for ffx.

    The returned mi array has a shape of (1, n_times)
    """
    # proper shape of the regressor
    n_times, _, n_trials = x.shape
    y_t = np.tile(y.reshape(1, 1, -1), (n_times, 1, 1))
    # compute mi across subject
    mi = mi_nd_gg(x, y_t, **CONFIG["KW_GCMI"])[np.newaxis, :]

    return mi


def mi_gg_rfx(x, y, z, suj):
    """I(C; C) for rfx.

    The returned mi array has a shape of (n_subjects, n_times)
    """
    # proper shape of the regressor
    n_times, _, n_trials = x.shape
    y_t = np.tile(y.reshape(1, 1, -1), (n_times, 1, 1))
    # get subject informations
    suj_u = np.unique(suj)
    n_subjects = len(suj_u)
    # compute mi per subject
    mi = np.zeros((n_subjects, n_times), dtype=float)
    for n_s, s in enumerate(suj_u):
        is_suj = suj == s
        mi[n_s, :] = mi_nd_gg(x[..., is_suj], y_t[..., is_suj],
                              **CONFIG["KW_GCMI"])

    return mi

###############################################################################
#                        I(CONTINUOUS; DISCRET)
###############################################################################


def mi_gd_ffx():
    """I(C; D) for ffx."""
    pass


def mi_gd_rfx():
    """I(C; D) for rfx."""
    pass

###############################################################################
#                        I(CONTINUOUS; CONTINUOUS | DISCRET)
###############################################################################


def mi_ggd_ffx():
    """I(C; C | D) for ffx."""
    pass


def mi_ggd_rfx():
    """I(C; C | D) for rfx."""
    pass


MI_FUN = dict(
    cc=dict(ffx=mi_gg_ffx, rfx=mi_gg_rfx),
    cd=dict(ffx=mi_gd_ffx, rfx=mi_gd_rfx),
    ccd=dict(ffx=mi_ggd_ffx, rfx=mi_ggd_rfx)
)

###############################################################################
#                              PERMUTATION VECTOR
###############################################################################


def permute_mi_vector(y, suj, mi_type='cc', inference='rfx', n_perm=1000):
    """Permute regressor variable for performing non-parameteric statistics.

    Parameters
    ----------
    y : array_like
        Array of shape (n_epochs,) to be permuted
    suj : array_like
        Array of shape (n_epochs,) used for permuting per subject
    mi_type : {'cc', 'cd', 'ccd'}
        Mutual information type
    inference : {'ffx', 'rfx'}
        Inference type (fixed or random effect)
    n_perm : int | 1000
        Number of permutations to return

    Returns
    -------
    y_p : list
        List of length (n_perm,) of random permutation of the regressor
    """
    y_p = []
    for p in range(n_perm):
        if inference == 'ffx':    # FFX (FIXED EFFECT)
            if mi_type in ['cc', 'cd']:
                y_p += [np.random.permutation(y)]
            elif mi_type == 'ccd':
                pass
        elif inference == 'rfx':  # RFX (RANDOM EFFECT)
            y_c = y.copy()
            for s in np.unique(suj):
                # find everywhere the subject is present
                is_suj = suj == s
                # randomize per subject
                if mi_type in ['cc', 'cd']:
                    y_c[is_suj] = np.random.permutation(y[is_suj])
                elif mi_type == 'ccd':
                    raise NotImplementedError("TODO NEXT")
            y_p += [y_c]
    assert len(y_p) == n_perm

    return y_p
