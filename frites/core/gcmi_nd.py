"""Multi-dimentional Gaussian copula mutual information estimation."""
import numpy as np
from scipy.special import psi
from itertools import product

from frites.core.copnorm import copnorm_nd

###############################################################################
###############################################################################
#                                 N-D TOOLS
###############################################################################
###############################################################################


def nd_reshape(x, mvaxis=None, traxis=-1):
    """Multi-dimentional reshaping.

    This function is used to be sure that an nd array has a correct shape
    of (..., mvaxis, traxis).

    Parameters
    ----------
    x : array_like
        Multi-dimentional array
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered

    Returns
    -------
    x_rsh : array_like
        The reshaped multi-dimentional array of shape (..., mvaxis, traxis)
    """
    assert isinstance(traxis, int)
    traxis = np.arange(x.ndim)[traxis]

    # Create an empty mvaxis axis
    if not isinstance(mvaxis, int):
        x = x[..., np.newaxis]
        mvaxis = -1
    assert isinstance(mvaxis, int)
    mvaxis = np.arange(x.ndim)[mvaxis]

    # move the multi-variate and trial axis
    x = np.moveaxis(x, (mvaxis, traxis), (-2, -1))

    return x


def nd_shape_checking(x, y, mvaxis, traxis):
    """Check that the shape between two ndarray is consitent.

    x.shape = (nx_1, ..., n_xn, x_mvaxis, traxis)
    y.shape = (nx_1, ..., n_xn, y_mvaxis, traxis)
    """
    assert x.ndim == y.ndim
    dims = np.delete(np.arange(x.ndim), -2)
    assert all([x.shape[k] == y.shape[k] for k in dims])


###############################################################################
###############################################################################
#                          MUTUAL INFORMATION
###############################################################################
###############################################################################


def mi_nd_gg(x, y, mvaxis=None, traxis=-1, biascorrect=True, demeaned=False,
             shape_checking=True):
    """Multi-dimentional MI between two Gaussian variables in bits.

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have the same shape except on the mvaxis
        (if needed).
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False

    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x and y, without the
        mvaxis and traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
        y = nd_reshape(y, mvaxis=mvaxis, traxis=traxis)
        nd_shape_checking(x, y, mvaxis, traxis)

    # x.shape (..., x_mvaxis, traxis)
    # y.shape (..., y_mvaxis, traxis)
    ntrl = x.shape[-1]
    nvarx, nvary = x.shape[-2], y.shape[-2]
    nvarxy = nvarx + nvary

    # joint variable along the mvaxis
    xy = np.concatenate((x, y), axis=-2)
    if not demeaned:
        xy -= xy.mean(axis=-1, keepdims=True)
    cxy = np.einsum('...ij, ...kj->...ik', xy, xy)
    cxy /= float(ntrl - 1.)

    # submatrices of joint covariance
    cx = cxy[..., :nvarx, :nvarx]
    cy = cxy[..., nvarx:, nvarx:]

    # Cholesky decomposition
    chcxy = np.linalg.cholesky(cxy)
    chcx = np.linalg.cholesky(cx)
    chcy = np.linalg.cholesky(cy)

    # entropies in nats
    # normalizations cancel for mutual information
    hx = np.log(np.einsum('...ii->...i', chcx)).sum(-1)
    hy = np.log(np.einsum('...ii->...i', chcy)).sum(-1)
    hxy = np.log(np.einsum('...ii->...i', chcxy)).sum(-1)

    ln2 = np.log(2)
    if biascorrect:
        vec = np.arange(1, nvarxy + 1)
        psiterms = psi((ntrl - vec).astype(float) / 2.0) / 2.0
        dterm = (ln2 - np.log(ntrl - 1.0)) / 2.0
        hx = hx - nvarx * dterm - psiterms[:nvarx].sum()
        hy = hy - nvary * dterm - psiterms[:nvary].sum()
        hxy = hxy - nvarxy * dterm - psiterms[:nvarxy].sum()

    # MI in bits
    i = (hx + hy - hxy) / ln2
    return i


def mi_model_nd_gd(x, y, mvaxis=None, traxis=-1, biascorrect=True,
                   demeaned=False, shape_checking=True):
    """Multi-dimentional MI between a Gaussian and a discret variables in bits.

    This function is based on ANOVA style model comparison.

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have the same shape except on the mvaxis
        (if needed).
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False

    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x and y, without the
        mvaxis and traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
    assert isinstance(y, np.ndarray) and (y.ndim == 1)
    assert x.shape[-1] == len(y)

    # x.shape (..., x_mvaxis, traxis)
    nvarx, ntrl = x.shape[-2], x.shape[-1]
    u_y = np.unique(y)
    sh = x.shape[:-2]
    zm_shape = list(sh) + [len(u_y)]

    # joint variable along the mvaxis
    if not demeaned:
        x = x - x.mean(axis=-1, keepdims=True)

    # class-conditional entropies
    ntrl_y = np.zeros((len(u_y),), dtype=int)
    hcond = np.zeros(zm_shape, dtype=float)
    # c = .5 * (np.log(2. * np.pi) + 1)
    for num, yi in enumerate(u_y):
        idx = y == yi
        xm = x[..., idx]
        ntrl_y[num] = idx.sum()
        xm = xm - xm.mean(axis=-1, keepdims=True)
        cm = np.einsum('...ij, ...kj->...ik', xm, xm) / float(ntrl_y[num] - 1.)
        chcm = np.linalg.cholesky(cm)
        hcond[..., num] = np.log(np.einsum('...ii->...i', chcm)).sum(-1)

    # class weights
    w = ntrl_y / float(ntrl)

    # unconditional entropy from unconditional Gaussian fit
    cx = np.einsum('...ij, ...kj->...ik', x, x) / float(ntrl - 1.)
    chc = np.linalg.cholesky(cx)
    hunc = np.log(np.einsum('...ii->...i', chc)).sum(-1)

    ln2 = np.log(2)
    if biascorrect:
        vars = np.arange(1, nvarx + 1)

        psiterms = psi((ntrl - vars).astype(float) / 2.) / 2.
        dterm = (ln2 - np.log(float(ntrl - 1))) / 2.
        hunc = hunc - nvarx * dterm - psiterms.sum()

        dterm = (ln2 - np.log((ntrl_y - 1).astype(float))) / 2.
        psiterms = np.zeros_like(ntrl_y, dtype=float)
        for vi in vars:
            idx = ntrl_y - vi
            psiterms = psiterms + psi(idx.astype(float) / 2.)
        hcond = hcond - nvarx * dterm - (psiterms / 2.)

    # MI in bits
    i = (hunc - np.einsum('i, ...i', w, hcond)) / ln2
    return i


def cmi_nd_ggg(x, y, z, mvaxis=None, traxis=-1, biascorrect=True,
               demeaned=False, shape_checking=True):
    """Multi-dimentional MI between three Gaussian variables in bits.

    This function is based on ANOVA style model comparison.

    Parameters
    ----------
    x, y, z : array_like
        Arrays to consider for computing the Mutual Information. The three
        input variables x, y and z should have the same shape except on the
        mvaxis (if needed).
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False

    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x, y and z without the
        mvaxis and traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
        y = nd_reshape(y, mvaxis=mvaxis, traxis=traxis)
        z = nd_reshape(z, mvaxis=mvaxis, traxis=traxis)
        nd_shape_checking(x, y, mvaxis, traxis)
        nd_shape_checking(x, z, mvaxis, traxis)

    # x.shape == y.shape == z.shape (..., x_mvaxis, traxis)
    ntrl = x.shape[-1]
    nvarx, nvary, nvarz = x.shape[-2], y.shape[-2], z.shape[-2]
    nvarxy = nvarx + nvary
    nvaryz = nvary + nvarz
    nvarxy = nvarx + nvary
    nvarxz = nvarx + nvarz
    nvarxyz = nvarx + nvaryz

    # joint variable along the mvaxis
    xyz = np.concatenate((x, y, z), axis=-2)
    if not demeaned:
        xyz -= xyz.mean(axis=-1, keepdims=True)
    cxyz = np.einsum('...ij, ...kj->...ik', xyz, xyz)
    cxyz /= float(ntrl - 1.)

    # submatrices of joint covariance
    cz = cxyz[..., nvarxy:, nvarxy:]
    cyz = cxyz[..., nvarx:, nvarx:]
    sh = list(cxyz.shape)
    sh[-1], sh[-2] = nvarxz, nvarxz
    cxz = np.zeros(tuple(sh), dtype=float)
    cxz[..., :nvarx, :nvarx] = cxyz[..., :nvarx, :nvarx]
    cxz[..., :nvarx, nvarx:] = cxyz[..., :nvarx, nvarxy:]
    cxz[..., nvarx:, :nvarx] = cxyz[..., nvarxy:, :nvarx]
    cxz[..., nvarx:, nvarx:] = cxyz[..., nvarxy:, nvarxy:]

    # Cholesky decomposition
    chcz = np.linalg.cholesky(cz)
    chcxz = np.linalg.cholesky(cxz)
    chcyz = np.linalg.cholesky(cyz)
    chcxyz = np.linalg.cholesky(cxyz)

    # entropies in nats
    # normalizations cancel for mutual information
    hz = np.log(np.einsum('...ii->...i', chcz)).sum(-1)
    hxz = np.log(np.einsum('...ii->...i', chcxz)).sum(-1)
    hyz = np.log(np.einsum('...ii->...i', chcyz)).sum(-1)
    hxyz = np.log(np.einsum('...ii->...i', chcxyz)).sum(-1)

    ln2 = np.log(2)
    if biascorrect:
        vec = np.arange(1, nvarxyz + 1)
        psiterms = psi((ntrl - vec).astype(float) / 2.0) / 2.0
        dterm = (ln2 - np.log(ntrl - 1.0)) / 2.0
        hz = hz - nvarz * dterm - psiterms[:nvarz].sum()
        hxz = hxz - nvarxz * dterm - psiterms[:nvarxz].sum()
        hyz = hyz - nvaryz * dterm - psiterms[:nvaryz].sum()
        hxyz = hxyz - nvarxyz * dterm - psiterms[:nvarxyz].sum()

    # MI in bits
    i = (hxz + hyz - hxyz - hz) / ln2
    return i


###############################################################################
###############################################################################
#                  GAUSSIAN COPULA MUTUAL INFORMATION
###############################################################################
###############################################################################


def gcmi_nd_cc(x, y, mvaxis=None, traxis=-1, shape_checking=True, gcrn=True):
    """GCMI between two continuous variables.

    The only difference with `mi_gg` is that a normalization is performed for
    each continuous variable.

    Parameters
    ----------
    x, y : array_like
        Continuous variables
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False
    gcrn : bool | True
        Apply a Gaussian Copula rank normalization. This operation is
        relatively slow for big arrays.

    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x and y, without the
        mvaxis and traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
        y = nd_reshape(y, mvaxis=mvaxis, traxis=traxis)
        nd_shape_checking(x, y, mvaxis, traxis)

    # x.shape (..., x_mvaxis, traxis)
    # y.shape (..., y_mvaxis, traxis)
    if gcrn:
        cx, cy = copnorm_nd(x, axis=-1), copnorm_nd(y, axis=-1)
    else:
        cx, cy = x, y
    return mi_nd_gg(cx, cy, mvaxis=-2, traxis=-1, biascorrect=True,
                    demeaned=True, shape_checking=False)


def gcmi_model_nd_cd(x, y, mvaxis=None, traxis=-1, shape_checking=True,
                     gcrn=True):
    """GCMI between a continuous and discret variables.

    The only difference with `mi_gg` is that a normalization is performed for
    each continuous variable.

    Parameters
    ----------
    x : array_like
        Continuous variable
    y : array_like
        Discret variable of shape (n_trials,)
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    shape_checking : bool | True
        Perform a reshape and check that x is consistents. For high
        performances and to avoid extensive memory usage, it's better to
        already have x with a shape of (..., mvaxis, traxis) and to set this
        parameter to False
    gcrn : bool | True
        Apply a Gaussian Copula rank normalization. This operation is
        relatively slow for big arrays.

    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x, without the mvaxis and
        traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)

    # x.shape (..., x_mvaxis, traxis)
    # y.shape (traxis)
    cx = copnorm_nd(x, axis=-1) if gcrn else x
    return mi_model_nd_gd(cx, y, mvaxis=-2, traxis=-1, biascorrect=True,
                          demeaned=True, shape_checking=False)

###############################################################################
###############################################################################
#               GAUSSIAN COPULA CONTIONAL MUTUAL INFORMATION
###############################################################################
###############################################################################


def gccmi_nd_ccnd(x, y, *z, mvaxis=None, traxis=-1, gcrn=True,
                  shape_checking=True, biascorrect=True, demeaned=True):
    """Conditional GCMI between two continuous variables.

    This function performs a GC-CMI between 2 continuous variables conditioned
    with multiple discrete variables.

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have the same shape except on the mvaxis
        (if needed).
    z : list | array_like
        Array that describes the conditions across the trial axis. Should be a
        list of arrays of shape (n_trials,) of integers
        (e.g. [0, 0, ..., 1, 1, 2, 2])
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    gcrn : bool | True
        Apply a Gaussian Copula rank normalization. This operation is
        relatively slow for big arrays.
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False

    Returns
    -------
    cmi : array_like
        Conditional mutual-information with the same shape as x and y without
        the mvaxis and traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
        y = nd_reshape(y, mvaxis=mvaxis, traxis=traxis)
        nd_shape_checking(x, y, mvaxis, traxis)
    ntrl = x.shape[-1]

    # Find unique values of each discret array
    prod_idx = discret_to_index(*z)
    # sh = x.shape[:-3] if isinstance(mvaxis, int) else x.shape[:-2]
    sh = x.shape[:-2]
    zm_shape = list(sh) + [len(prod_idx)]

    # calculate gcmi for each z value
    pz = np.zeros((len(prod_idx),), dtype=float)
    icond = np.zeros(zm_shape, dtype=float)
    for num, idx in enumerate(prod_idx):
        pz[num] = idx.sum()
        if gcrn:
            thsx = copnorm_nd(x[..., idx], axis=-1)
            thsy = copnorm_nd(y[..., idx], axis=-1)
        else:
            thsx = x[..., idx]
            thsy = y[..., idx]
        icond[..., num] = mi_nd_gg(thsx, thsy, mvaxis=-2, traxis=-1,
                                   biascorrect=biascorrect, demeaned=demeaned,
                                   shape_checking=False)
    pz /= ntrl

    # conditional mutual information
    cmi = np.sum(pz * icond, axis=-1)
    return cmi


def cmi_nd_ggd(x, y, z, mvaxis=None, traxis=-1, shape_checking=True,
               biascorrect=True, demeaned=False):
    """Conditional MI between a continuous and a discret variable.

    This function performs a CMI between a continuous and a discret variable
    conditioned with multiple discrete variables.

    Parameters
    ----------
    x : array_like
        Continuous variable
    y : array_like
        Discret variable
    z : list | array_like
        Array that describes the conditions across the trial axis of shape
        (n_trials,)
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)

    Returns
    -------
    cmi : array_like
        Conditional mutual-information with the same shape as x and y without
        the mvaxis and traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
        y = nd_reshape(y, mvaxis=mvaxis, traxis=traxis)
        nd_shape_checking(x, y, mvaxis, traxis)
        ntrl = x.shape[-1]
        assert (z.ndim == 1) and (len(z) == ntrl)
    ntrl = x.shape[-1]

    # sh = x.shape[:-3] if isinstance(mvaxis, int) else x.shape[:-2]
    u_z = np.unique(z)
    sh = x.shape[:-2]
    zm_shape = list(sh) + [len(u_z)]

    # calculate gcmi for each z value
    pz = np.zeros((len(u_z),), dtype=float)
    icond = np.zeros(zm_shape, dtype=float)
    for n_z, zi in enumerate(u_z):
        idx = z == zi
        pz[n_z] = idx.sum()
        thsx, thsy = x[..., idx], y[..., idx]
        icond[..., n_z] = mi_nd_gg(thsx, thsy, mvaxis=-2, traxis=-1,
                                   biascorrect=biascorrect, demeaned=demeaned,
                                   shape_checking=False)
    pz /= ntrl

    # conditional mutual information
    cmi = np.sum(np.multiply(pz, icond), axis=-1)
    return cmi


def gccmi_model_nd_cdnd(x, y, *z, mvaxis=None, traxis=-1, gcrn=True,
                        shape_checking=True):
    """Conditional GCMI between a continuous and a discret variable.

    This function performs a GC-CMI between a continuous and a discret
    variable conditioned with multiple discrete variables.

    Parameters
    ----------
    x : array_like
        Continuous variable
    y : array_like
        Discret variable
    z : list | array_like
        Array that describes the conditions across the trial axis. Should be a
        list of arrays of shape (n_trials,) of integers
        (e.g. [0, 0, ..., 1, 1, 2, 2])
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    gcrn : bool | True
        Apply a Gaussian Copula rank normalization. This operation is
        relatively slow for big arrays.
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False

    Returns
    -------
    cmi : array_like
        Conditional mutual-information with the same shape as x and y without
        the mvaxis and traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
    assert isinstance(y, np.ndarray) and (y.ndim == 1)
    assert x.shape[-1] == len(y)
    ntrl = x.shape[-1]

    # Find unique values of each discret array
    prod_idx = discret_to_index(*z)
    # sh = x.shape[:-3] if isinstance(mvaxis, int) else x.shape[:-2]
    sh = x.shape[:-2]
    zm_shape = list(sh) + [len(prod_idx)]

    # calculate gcmi for each z value
    pz = np.zeros((len(prod_idx),), dtype=float)
    icond = np.zeros(zm_shape, dtype=float)
    for num, idx in enumerate(prod_idx):
        pz[num] = idx.sum()
        if gcrn:
            thsx = copnorm_nd(x[..., idx], axis=-1)
        else:
            thsx = x[..., idx]
        thsy = y[idx]
        icond[..., num] = mi_model_nd_gd(thsx, thsy, mvaxis=-2, traxis=-1,
                                         biascorrect=True, demeaned=True,
                                         shape_checking=False)
    pz /= ntrl

    # conditional mutual information
    cmi = np.sum(pz * icond, axis=-1)
    return cmi


def discret_to_index(*z):
    """Convert a list of discret variables into boolean indices.

    Parameters
    ----------
    z : tuple | list
        List of discret variables

    Returns
    -------
    idx : list
        List of boolean arrays. Each array specify the condition to use
    """
    if isinstance(z, np.ndarray) and (z.ndim == 1):
        return [z == k for k in np.unique(z)]
    elif isinstance(z, (tuple, list)):
        # array checking
        is_array = all([isinstance(k, np.ndarray) for k in z])
        is_vec = all([k.ndim == 1 for k in z])
        is_shape = all([z[0].shape == k.shape for k in z])
        if not (is_array and is_vec and is_shape):
            raise TypeError("z should be a list of 1-D array, all with the "
                            "same shape")

        # build unique indices
        u_z = tuple([tuple(np.unique(k)) for k in z])
        idx = []
        for k in product(*u_z):
            _idx = []
            for _c, _k in zip(z, k):
                _idx += [_c == _k]
            _idx_bool = np.all(np.c_[_idx], axis=0)
            if _idx_bool.any():
                idx += [_idx_bool]
        return idx


def gccmi_nd_ccc(x, y, z, mvaxis=None, traxis=-1, shape_checking=True,
                 gcrn=True):
    """GCCMI between two continuous variables conditioned on a third.

    Parameters
    ----------
    x, y, z : array_like
        Continuous variables. z is the continuous variable that is considered
        as the condition
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False
    gcrn : bool | True
        Apply a Gaussian Copula rank normalization. This operation is
        relatively slow for big arrays.

    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x and y, without the
        mvaxis and traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
        y = nd_reshape(y, mvaxis=mvaxis, traxis=traxis)
        z = nd_reshape(z, mvaxis=mvaxis, traxis=traxis)
        nd_shape_checking(x, y, mvaxis, traxis)
        nd_shape_checking(x, z, mvaxis, traxis)

    # x.shape == y.shape == z.shape (..., x_mvaxis, traxis)
    if gcrn:
        cx, cy = copnorm_nd(x, axis=-1), copnorm_nd(y, axis=-1)
        cz = copnorm_nd(z, axis=-1)
    else:
        cx, cy, cz = x, y, z
    return cmi_nd_ggg(cx, cy, cz, mvaxis=-2, traxis=-1, biascorrect=True,
                      demeaned=True, shape_checking=False)
