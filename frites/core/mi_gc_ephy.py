"""High-level MI functions for ffx / rfx inferences.

The functions in this file are used to compute the mutual information on
electrophysiological data organized as follow (n_times, 1, n_trials).
"""
import numpy as np

from frites.config import CONFIG
from frites.core import mi_nd_gg, mi_model_nd_gd, gccmi_nd_ccnd


###############################################################################
###############################################################################
#                        I(CONTINUOUS; CONTINUOUS)
###############################################################################
###############################################################################


def mi_gc_ephy_cc(x, y, z, suj, inference, **kwargs):
    """I(C; C) for rfx.

    The returned mi array has a shape of (n_subjects, n_times) if inference is
    "rfx", (1, n_times) if "ffx".
    """
    # proper shape of the regressor
    n_times, _, n_trials = x.shape
    y_t = np.tile(y.T[np.newaxis, ...], (n_times, 1, 1))
    # compute mi across (ffx) or per subject (rfx)
    if inference == 'ffx':
        mi = mi_nd_gg(x, y_t, **CONFIG["KW_GCMI"])[np.newaxis, :]
    elif inference == 'rfx':
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


def mi_gc_ephy_conn_cc(x_1, x_2, suj_1, suj_2, inference, **kwargs):
    """I(C; C) for rfx.

    The returned mi array has a shape of (n_subjects, n_times) if inference is
    "rfx", (1, n_times) if "ffx".
    """
    # proper shape of the regressor
    n_times, _, n_trials = x_1.shape
    # compute mi across (ffx) or per subject (rfx)
    if inference == 'ffx':
        mi = mi_nd_gg(x_1, x_2, **CONFIG["KW_GCMI"])[np.newaxis, :]
    elif inference == 'rfx':
        # get subject informations
        suj_u = np.intersect1d(suj_1, suj_2)
        n_subjects = len(suj_u)
        # compute mi per subject
        mi = np.zeros((n_subjects, n_times), dtype=float)
        for n_s, s in enumerate(suj_u):
            is_suj_1 = suj_1 == s
            is_suj_2 = suj_2 == s
            mi[n_s, :] = mi_nd_gg(x_1[..., is_suj_1], x_2[..., is_suj_2],
                                  **CONFIG["KW_GCMI"])

    return mi

###############################################################################
###############################################################################
#                        I(CONTINUOUS; DISCRET)
###############################################################################
###############################################################################


def mi_gc_ephy_cd(x, y, z, suj, inference, **kwargs):
    """I(C; D) for ffx.

    The returned mi array has a shape of (n_subjects, n_times) if inference is
    "rfx", (1, n_times) if "ffx".
    """
    n_times, _, _ = x.shape
    _y = y.squeeze().astype(int)
    # compute mi across (ffx) or per subject (rfx)
    if inference == 'ffx':
        mi = mi_model_nd_gd(x, _y, **CONFIG["KW_GCMI"])[np.newaxis, :]
    elif inference == 'rfx':
        # get subject informations
        suj_u = np.unique(suj)
        n_subjects = len(suj_u)
        # compute mi per subject
        mi = np.zeros((n_subjects, n_times), dtype=float)
        for n_s, s in enumerate(suj_u):
            is_suj = suj == s
            mi[n_s, :] = mi_model_nd_gd(x[..., is_suj], _y[is_suj],
                                        **CONFIG["KW_GCMI"])

    return mi


###############################################################################
###############################################################################
#                        I(CONTINUOUS; CONTINUOUS | DISCRET)
###############################################################################
###############################################################################


def mi_gc_ephy_ccd(x, y, z, suj, inference, **kwargs):
    """I(C; C | D) for ffx.

    The returned mi array has a shape of (n_subjects, n_times) if inference is
    "rfx", (1, n_times) if "ffx".
    """
    # discard gcrn
    kw = CONFIG["KW_GCMI"].copy()
    kw['gcrn'] = False
    # proper shape of the regressor
    n_times, _, n_trials = x.shape
    y_t = np.tile(y.T[np.newaxis, ...], (n_times, 1, 1))
    # compute mi across (ffx) or per subject (rfx)
    if inference == 'ffx':
        _z = tuple([z[:, n] for n in range(z.shape[1])])
        mi = gccmi_nd_ccnd(x, y_t, *_z, **kw)[np.newaxis, :]
    elif inference == 'rfx':
        # get subject informations
        suj_u = np.unique(suj)
        n_subjects = len(suj_u)
        # compute mi per subject
        mi = np.zeros((n_subjects, n_times), dtype=float)
        for n_s, s in enumerate(suj_u):
            is_suj = suj == s
            _z = tuple([z[is_suj, n] for n in range(z.shape[1])])
            mi[n_s, :] = gccmi_nd_ccnd(x[..., is_suj], y_t[..., is_suj], *_z,
                                       **kw)

    return mi
