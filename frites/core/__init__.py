"""Core functions of information theoretical measures.

This submodule contains all of the functions to compute mutual-information,
entropy etc..

For further details about the Gaussian-Copula Mutual information see
:cite:`ince2017statistical`
"""
from .copnorm import (copnorm_1d, copnorm_cat_1d, copnorm_nd, copnorm_cat_nd)  # noqa
from .gcmi_1d import (ent_1d_g, mi_1d_gg, gcmi_1d_cc, mi_model_1d_gd,  # noqa
                      gcmi_model_1d_cd, mi_mixture_1d_gd, gcmi_mixture_1d_cd,
                      cmi_1d_ggg, gccmi_1d_ccc, gccmi_1d_ccd)
from .gcmi_nd import (mi_nd_gg, mi_model_nd_gd, cmi_nd_ggg, gcmi_nd_cc,  # noqa
                      gcmi_model_nd_cd, gccmi_nd_ccnd, gccmi_model_nd_cdnd,
                      gccmi_nd_ccc)
from .mi_stats import (permute_mi_vector) # noqa
from .it import (it_transfer_entropy, it_fit)  # noqa

# -----------------------------------------------------------------------------
# get the core functions to use for the mi estimation


def get_core_mi_fun(mi_method):
    """Get the core functions for estimating the mutual information.

    Parameters
    ----------
    mi_method : {'gc', 'bin'}
        Method type for computing MI. Use either :

            * 'gc' : Gaussian-Copula based methods
            * 'bin' : binning based methods

    Returns
    -------
    mi_fun : dict
        Dictionary of methods
    """
    assert mi_method in ['gc', 'bin']
    if mi_method is 'gc':
        from .mi_gc_ephy import (mi_gc_ephy_cc, mi_gc_ephy_cd, mi_gc_ephy_ccd)
        mi_fun = dict(cc=mi_gc_ephy_cc, cd=mi_gc_ephy_cd, ccd=mi_gc_ephy_ccd)
    elif mi_method is 'bin':
        from .mi_bin_ephy import (mi_bin_ephy_cc, mi_bin_ephy_cd,
                                  mi_bin_ephy_ccd)
        mi_fun = dict(cc=mi_bin_ephy_cc, cd=mi_bin_ephy_cd,
                      ccd=mi_bin_ephy_ccd)
    return mi_fun
