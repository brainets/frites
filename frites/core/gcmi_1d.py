"""
Gaussian copula mutual information estimation.

| **Authors** : Robin AA. Ince
| **Original code** : https://github.com/robince/gcmi
| **Reference** :
| RAA Ince, BL Giordano, C Kayser, GA Rousselet, J Gross and PG Schyns
"A statistical framework for neuroimaging data analysis based on mutual
information estimated via a Gaussian copula" Human Brain Mapping (2017)
38 p. 1541-1573 doi:10.1002/hbm.23471
"""
import numpy as np
import scipy as sp

from frites.core import copnorm_nd, copnorm_cat_nd


def ent_1d_g(x, biascorrect=True):
    """Entropy of a Gaussian variable in bits.

    H = ent_g(x) returns the entropy of a (possibly multidimensional) Gaussian
    variable x with bias correction.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_epochs,)
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI

    Returns
    -------
    hx : float
        Entropy of the gaussian variable (in bits)
    """
    x = np.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")  # noqa
    nvarx, ntrl = x.shape

    # demean data
    x = x - x.mean(axis=1)[:, np.newaxis]
    # covariance
    c = np.dot(x, x.T) / float(ntrl - 1)
    chc = np.linalg.cholesky(c)

    # entropy in nats
    hx = np.sum(np.log(np.diagonal(chc))) + .5 * nvarx * (
        np.log(2 * np.pi) + 1.)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi((ntrl - np.arange(1, nvarx + 1).astype(
            float)) / 2.) / 2.
        dterm = (ln2 - np.log(ntrl - 1.)) / 2.
        hx = hx - nvarx * dterm - psiterms.sum()

    # convert to bits
    return hx / ln2


def mi_1d_gg(x, y, biascorrect=True, demeaned=False):
    """Mutual information (MI) between two Gaussian variables in bits.

    I = mi_gg(x,y) returns the MI between two (possibly multidimensional)
    Gaussian variables, x and y, with bias correction.

    Parameters
    ----------
    x, y : array_like
        Gaussian arrays of shape (n_epochs,) or (n_dimensions, n_epochs)
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)

    Returns
    -------
    i : float
        Information shared by x and y (in bits)
    """
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    if (x.ndim > 2) or (y.ndim > 2):
        raise ValueError("x and y must be at most 2d")
    nvarx, ntrl = x.shape
    nvary = y.shape[0]
    nvarxy = nvarx + nvary

    if y.shape[1] != ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xy = np.vstack((x, y))
    if not demeaned:
        xy = xy - xy.mean(axis=1)[:, np.newaxis]
    cxy = np.dot(xy, xy.T) / float(ntrl - 1)
    # submatrices of joint covariance
    cx = cxy[:nvarx, :nvarx]
    cy = cxy[nvarx:, nvarx:]

    chcxy = np.linalg.cholesky(cxy)
    chcx = np.linalg.cholesky(cx)
    chcy = np.linalg.cholesky(cy)

    # entropies in nats
    # normalizations cancel for mutual information
    hx = np.sum(np.log(np.diagonal(chcx)))
    hy = np.sum(np.log(np.diagonal(chcy)))
    hxy = np.sum(np.log(np.diagonal(chcxy)))

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi(
            (ntrl - np.arange(1, nvarxy + 1)).astype(float) / 2.) / 2.
        dterm = (ln2 - np.log(ntrl - 1.)) / 2.
        hx = hx - nvarx * dterm - psiterms[:nvarx].sum()
        hy = hy - nvary * dterm - psiterms[:nvary].sum()
        hxy = hxy - nvarxy * dterm - psiterms[:nvarxy].sum()

    # MI in bits
    i = (hx + hy - hxy) / ln2
    return i


def gcmi_1d_cc(x, y):
    """Gaussian-Copula MI between two continuous variables.

    I = gcmi_cc(x,y) returns the MI between two (possibly multidimensional)
    continuous variables, x and y, estimated via a Gaussian copula.

    Parameters
    ----------
    x, y : array_like
        Continuous arrays of shape (n_epochs,) or (n_dimensions, n_epochs)

    Returns
    -------
    i : float
        Information shared by x and y (in bits)
    """
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    nvarx, ntrl = x.shape

    if y.shape[1] != ntrl:
        raise ValueError("number of trials do not match")

    # copula normalization
    cx, cy = copnorm_nd(x, axis=1), copnorm_nd(y, axis=1)
    # parametric Gaussian MI
    return mi_1d_gg(cx, cy, True, True)


def mi_model_1d_gd(x, y, biascorrect=True, demeaned=False):
    """Mutual information between a Gaussian and a discrete variable in bits.

    This method is based on ANOVA style model comparison.
    I = mi_model_gd(x,y) returns the MI between the (possibly multidimensional)
    Gaussian variable x and the discrete variable y.

    Parameters
    ----------
    x, y : array_like
        Gaussian arrays of shape (n_epochs,) or (n_dimensions, n_epochs). y
        must be an array of integers
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)

    Returns
    -------
    i : float
        Information shared by x and y (in bits)
    """
    x, y = np.atleast_2d(x), np.squeeze(y)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    if y.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(y.dtype, int):
        raise ValueError("y should be an integer array")

    nvarx, ntrl = x.shape
    ym = np.unique(y)

    if y.size != ntrl:
        raise ValueError("number of trials do not match")

    if not demeaned:
        x = x - x.mean(axis=1)[:, np.newaxis]

    # class-conditional entropies
    ntrl_y = np.zeros(len(ym))
    hcond = np.zeros(len(ym))
    for n_yi, yi in enumerate(ym):
        idx = y == yi
        xm = x[:, idx]
        ntrl_y[n_yi] = xm.shape[1]
        xm = xm - xm.mean(axis=1)[:, np.newaxis]
        cm = np.dot(xm, xm.T) / float(ntrl_y[n_yi] - 1)
        chcm = np.linalg.cholesky(cm)
        hcond[n_yi] = np.sum(np.log(np.diagonal(chcm)))

    # class weights
    w = ntrl_y / float(ntrl)

    # unconditional entropy from unconditional Gaussian fit
    cx = np.dot(x, x.T) / float(ntrl - 1)
    chc = np.linalg.cholesky(cx)
    hunc = np.sum(np.log(np.diagonal(chc)))  # + c*nvarx

    ln2 = np.log(2)
    if biascorrect:
        vars = np.arange(1, nvarx + 1)

        psiterms = sp.special.psi((ntrl - vars).astype(float) / 2.) / 2.
        dterm = (ln2 - np.log(float(ntrl - 1))) / 2.
        hunc = hunc - nvarx * dterm - psiterms.sum()

        dterm = (ln2 - np.log((ntrl_y - 1).astype(float))) / 2.0
        psiterms = np.zeros(len(ym))
        for vi in vars:
            idx = ntrl_y - vi
            psiterms = psiterms + sp.special.psi(idx.astype(float) / 2.)
        hcond = hcond - nvarx * dterm - (psiterms / 2.)

    # MI in bits
    i = (hunc - np.sum(w * hcond)) / ln2
    return i


def gcmi_model_1d_cd(x, y):
    """Gaussian-Copula MI between a continuous and a discrete variable.

    This method is based on ANOVA style model comparison.
    I = gcmi_model_cd(x,y,Ym) returns the MI between the (possibly
    multidimensional) continuous variable x and the discrete variable y.

    Parameters
    ----------
    x, y : array_like
        Continuous arrays of shape (n_epochs,) or (n_dimensions, n_epochs). y
        must be an array of integers

    Returns
    -------
    i : float
        Information shared by x and y (in bits)
    """
    x, y = np.atleast_2d(x), np.squeeze(y)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    if y.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(y.dtype, int):
        raise ValueError("y should be an integer array")

    nvarx, ntrl = x.shape

    if y.size != ntrl:
        raise ValueError("number of trials do not match")

    # copula normalization
    cx = copnorm_nd(x, axis=1)
    # parametric Gaussian MI
    return mi_model_1d_gd(cx, y, True, True)


def mi_mixture_1d_gd(x, y):
    """Mutual information between a Gaussian and a discrete variable in bits.

    This method evaluate MI from a Gaussian mixture.
    I = mi_mixture_gd(x,y) returns the MI between the (possibly
    multidimensional)

    Parameters
    ----------
    x, y : array_like
        Gaussian arrays of shape (n_epochs,) or (n_dimensions, n_epochs). y
        must be an array of integers

    Returns
    -------
    i : float
        Information shared by x and y (in bits)
    """
    x, y = np.atleast_2d(x), np.squeeze(y)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    if y.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(y.dtype, int):
        raise ValueError("y should be an integer array")

    nvarx, ntrl = x.shape
    ym = np.unique(y)

    if y.size != ntrl:
        raise ValueError("number of trials do not match")

    # class-conditional entropies
    ntrl_y = np.zeros((len(ym),))
    hcond = np.zeros((len(ym),))
    m = np.zeros((len(ym), nvarx))
    w = np.zeros((len(ym),))
    cc = .5 * (np.log(2. * np.pi) + 1)
    c = np.zeros((len(ym), nvarx, nvarx))
    chc = np.zeros((len(ym), nvarx, nvarx))
    for n_yi, yi in enumerate(ym):
        # class conditional data
        idx = y == yi
        xm = x[:, idx]
        # class mean
        m[n_yi, :] = xm.mean(axis=1)
        ntrl_y[n_yi] = xm.shape[1]

        xm = xm - m[n_yi, :][:, np.newaxis]
        c[n_yi, :, :] = np.dot(xm, xm.T) / float(ntrl_y[n_yi] - 1)
        chc[n_yi, :, :] = np.linalg.cholesky(c[n_yi, :, :])
        hcond[n_yi] = np.sum(np.log(np.diagonal(chc[n_yi, :, :]))) + cc * nvarx

    # class weights
    w = ntrl_y / float(ntrl)

    # mixture entropy via unscented transform
    # See:
    # Huber, Bailey, Durrant-Whyte and Hanebeck
    # "On entropy approximation for Gaussian mixture random vectors"
    # http://dx.doi.org/10.1109/MFI.2008.4648062

    # Goldberger, Gordon, Greenspan
    # "An efficient image similarity measure based on approximations of
    # KL-divergence between two Gaussian mixtures"
    # http://dx.doi.org/10.1109/ICCV.2003.1238387
    d = nvarx
    ds = np.sqrt(nvarx)
    hmix = 0.0
    for yi in range(len(ym)):
        ps = ds * chc[yi, :, :].T
        thsm = m[yi, :, np.newaxis]
        # unscented points for this class
        usc = np.hstack([thsm + ps, thsm - ps])

        # class log-likelihoods at unscented points
        log_lik = np.zeros((len(ym), 2 * nvarx))
        for mi in range(len(ym)):
            # demean points
            dx = usc - m[mi, :, np.newaxis]
            # gaussian likelihood
            log_lik[mi, :] = _norm_innerv(
                dx, chc[mi, :, :]) - hcond[mi] + .5 * nvarx

        # log mixture likelihood for these unscented points
        # sum over classes, axis=0
        # logmixlik = sp.misc.logsumexp(log_lik, axis=0, b=w[:, np.newaxis])
        logmixlik = np.log(np.sum(w[:, np.newaxis] * np.exp(log_lik)))

        # add to entropy estimate (sum over unscented points for this class)
        hmix = hmix + w[yi] * logmixlik.sum()

    hmix = -hmix / (2 * d)

    # no bias correct
    i = (hmix - np.sum(w * hcond)) / np.log(2.)
    return i


def _norm_innerv(x, chc):
    """Normalised innervations."""
    m = np.linalg.solve(chc, x)
    w = -0.5 * (m * m).sum(axis=0)
    return w


def gcmi_mixture_1d_cd(x, y):
    """Gaussian-Copula MI between a continuous and a discrete variable.

    This method evaluate MI from a Gaussian mixture.

    The Gaussian mixture is fit using robust measures of location (median) and
    scale (median absolute deviation) for each class.
    I = gcmi_mixture_cd(x,y) returns the MI between the (possibly
    multidimensional).

    Parameters
    ----------
    x, y : array_like
        Continuous arrays of shape (n_epochs,) or (n_dimensions, n_epochs). y
        must be an array of integers

    Returns
    -------
    i : float
        Information shared by x and y (in bits)
    """
    x, y = np.atleast_2d(x), np.squeeze(y)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    if y.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(y.dtype, int):
        raise ValueError("y should be an integer array")

    nvarx, ntrl = x.shape
    ym = np.unique(y)

    if y.size != ntrl:
        raise ValueError("number of trials do not match")

    # copula normalise each class
    # shift and rescale to match loc and scale of raw data
    # this provides a robust way to fit the gaussian mixture
    classdat = []
    ydat = []
    for yi in ym:
        # class conditional data
        idx = y == yi
        xm = x[:, idx]
        cxm = copnorm_nd(xm, axis=1)

        xmmed = np.median(xm, axis=1)[:, np.newaxis]
        # robust measure of s.d. under Gaussian assumption from median
        # absolute deviation
        xmmad = np.median(np.abs(xm - xmmed), axis=1)[:, np.newaxis]
        cxmscaled = cxm * (1.482602218505602 * xmmad)
        # robust measure of loc from median
        cxmscaled = cxmscaled + xmmed
        classdat.append(cxmscaled)
        ydat.append(yi * np.ones(xm.shape[1], dtype=int))

    cx = np.concatenate(classdat, axis=1)
    newy = np.concatenate(ydat)
    return mi_mixture_1d_gd(cx, newy)


def cmi_1d_ggg(x, y, z, biascorrect=True, demeaned=False):
    """Conditional MI between two Gaussian variables conditioned on a third.

    I = cmi_ggg(x,y,z) returns the CMI between two (possibly multidimensional)
    Gaussian variables, x and y, conditioned on a third, z, with bias
    correction.

    Parameters
    ----------
    x, y, z : array_like
        Gaussians arrays of shape (n_epochs,) or (n_dimensions, n_epochs).
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)

    Returns
    -------
    i : float
        Information shared by x and y conditioned by z (in bits)
    """
    x, y, z = np.atleast_2d(x), np.atleast_2d(y), np.atleast_2d(z)
    if x.ndim > 2 or y.ndim > 2 or z.ndim > 2:
        raise ValueError("x, y and z must be at most 2d")
    ntrl = x.shape[1]
    nvarx = x.shape[0]
    nvary = y.shape[0]
    nvarz = z.shape[0]
    nvaryz = nvary + nvarz
    nvarxy = nvarx + nvary
    nvarxz = nvarx + nvarz
    nvarxyz = nvarx + nvaryz

    if y.shape[1] != ntrl or z.shape[1] != ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xyz = np.vstack((x, y, z))
    if not demeaned:
        xyz = xyz - xyz.mean(axis=1)[:, np.newaxis]
    cxyz = np.dot(xyz, xyz.T) / float(ntrl - 1)
    # submatrices of joint covariance
    cz = cxyz[nvarxy:, nvarxy:]
    cyz = cxyz[nvarx:, nvarx:]
    cxz = np.zeros((nvarxz, nvarxz))
    cxz[:nvarx, :nvarx] = cxyz[:nvarx, :nvarx]
    cxz[:nvarx, nvarx:] = cxyz[:nvarx, nvarxy:]
    cxz[nvarx:, :nvarx] = cxyz[nvarxy:, :nvarx]
    cxz[nvarx:, nvarx:] = cxyz[nvarxy:, nvarxy:]

    chcz = np.linalg.cholesky(cz)
    chcxz = np.linalg.cholesky(cxz)
    chcyz = np.linalg.cholesky(cyz)
    chcxyz = np.linalg.cholesky(cxyz)

    # entropies in nats
    # normalizations cancel for cmi
    hz = np.sum(np.log(np.diagonal(chcz)))
    hxz = np.sum(np.log(np.diagonal(chcxz)))
    hyz = np.sum(np.log(np.diagonal(chcyz)))
    hxyz = np.sum(np.log(np.diagonal(chcxyz)))

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi(
            (ntrl - np.arange(1, nvarxyz + 1)).astype(float) / 2.) / 2.
        dterm = (ln2 - np.log(ntrl - 1.)) / 2.
        hz = hz - nvarz * dterm - psiterms[:nvarz].sum()
        hxz = hxz - nvarxz * dterm - psiterms[:nvarxz].sum()
        hyz = hyz - nvaryz * dterm - psiterms[:nvaryz].sum()
        hxyz = hxyz - nvarxyz * dterm - psiterms[:nvarxyz].sum()

    # MI in bits
    i = (hxz + hyz - hxyz - hz) / ln2
    return i


def gccmi_1d_ccc(x, y, z, biascorrect=True):
    """Gaussian-Copula CMI between three continuous variables.

    I = gccmi_1d_ccc(x,y,z) returns the CMI between two (possibly
    multidimensional) continuous variables, x and y, conditioned on a third, z,
    estimated via a Gaussian copula.

    Parameters
    ----------
    x, y, z : array_like
        Continuous arrays of shape (n_epochs,) or (n_dimensions, n_epochs).

    Returns
    -------
    i : float
        Information shared by x and y conditioned by z (in bits)
    """
    x, y, z = np.atleast_2d(x), np.atleast_2d(y), np.atleast_2d(z)
    if x.ndim > 2 or y.ndim > 2 or z.ndim > 2:
        raise ValueError("x, y and z must be at most 2d")

    nvarx, ntrl = x.shape

    if y.shape[1] != ntrl or z.shape[1] != ntrl:
        raise ValueError("number of trials do not match")

    # copula normalization
    cx = copnorm_nd(x, axis=1)
    cy = copnorm_nd(y, axis=1)
    cz = copnorm_nd(z, axis=1)
    # parametric Gaussian CMI
    return cmi_1d_ggg(cx, cy, cz, biascorrect=True, demeaned=True)


def cmi_1d_ggd(x, y, z, biascorrect=True, demeaned=False):
    """MI between 2 continuous variables conditioned on a discrete variable.

    I = cmi_1d_ggd(x,y,z) returns the CMI between two (possibly
    multidimensional) continuous variables, x and y, conditioned on a third
    discrete variable z, estimated via a Gaussian copula.

    Parameters
    ----------
    x, y : array_like
        Continuous arrays of shape (n_epochs,) or (n_dimensions, n_epochs).
    z : array_like
        Discret array of shape (n_epochs,)

    Returns
    -------
    cmi : float
        Conditional Mutual Information shared by x and y conditioned by z
        (in bits)
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    if z.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(z.dtype, int):
        raise ValueError("z should be an integer array")

    nvarx, ntrl = x.shape
    u_z = np.unique(z)

    if y.shape[1] != ntrl or z.size != ntrl:
        raise ValueError("number of trials do not match")

    # calculate gcmi for each z value
    icond = np.zeros((len(u_z),))
    pz = np.zeros((len(u_z),))
    for n_z, zi in enumerate(u_z):
        idx = z == zi
        thsx, thsy = x[:, idx], y[:, idx]
        pz[n_z] = idx.sum()
        icond[n_z] = mi_1d_gg(thsx, thsy, biascorrect=biascorrect,
                              demeaned=demeaned)

    pz /= float(ntrl)

    # conditional mutual information
    cmi = np.sum(pz * icond)
    return cmi


def gccmi_1d_ccd(x, y, z, biascorrect=True, demeaned=False):
    """GCCMI between 2 continuous variables conditioned on a discrete variable.

    I = gccmi_ccd(x,y,z) returns the CMI between two (possibly
    multidimensional) continuous variables, x and y, conditioned on a third
    discrete variable z, estimated via a Gaussian copula.

    Parameters
    ----------
    x, y : array_like
        Continuous arrays of shape (n_epochs,) or (n_dimensions, n_epochs).
    z : array_like
        Discret array of shape (n_epochs,)

    Returns
    -------
    cmi : float
        Conditional Mutual Information shared by x and y conditioned by z
        (in bits)
    """
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    x = copnorm_cat_nd(x, z, axis=-1)
    y = copnorm_cat_nd(y, z, axis=-1)
    return cmi_1d_ggd(x, y, z, biascorrect=biascorrect, demeaned=True)
