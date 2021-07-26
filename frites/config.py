"""Configuration file."""
import numpy as np
import mne

# Empty config
CONFIG = dict()

# Supported MNE types
MNE_EPOCHS_TYPE = (mne.Epochs, mne.EpochsArray, mne.epochs.EpochsFIF,
                   mne.epochs.BaseEpochs)
CONFIG["MNE_EPOCHS_TYPE"] = MNE_EPOCHS_TYPE
CONFIG["MNE_EPOCHSTFR_TYPE"] = (mne.time_frequency.EpochsTFR)

# Int and Float types
INT_DTYPE = (int, np.int8, np.int16, np.int32, np.int64)
FLOAT_DTYPE = (float, np.float16, np.float32, np.float64)
STR_DTYPE = (str, np.string_)
CONFIG['INT_DTYPE'] = INT_DTYPE
CONFIG['FLOAT_DTYPE'] = FLOAT_DTYPE
CONFIG['STR_DTYPE'] = STR_DTYPE

# gcmi configuration
CONFIG["KW_GCMI"] = dict(shape_checking=False, biascorrect=True,
                         demeaned=False, mvaxis=-2, traxis=-1)

# copula name conversion
CONFIG["COPULA_CONV"] = dict(cc='gg', cd='gd', ccd='ggd')

# mi types table
CONFIG['MI_TABLE'] = {
    'int': {
        'none': 'cd',
        'int': 'cd',
        'float': 'none'
    },
    'float': {
        'none': 'cc',
        'int': 'ccd',
        'float': 'ccc'
    },
    'none': {
        'none': 'none',
        'int': 'none',
        'float': 'none',
    }
}

# mi type full description
CONFIG['MI_REPR'] = {
    'none': 'none',
    'cc': 'I(x; y (continuous))',
    'cd': 'I(x; y (discret))',
    'ccd': 'I(x; y (continuous)) | z (discret)',
    'ccc': 'I(x; y (continuous)) | z (continuous)',
}

# general joblib config
CONFIG["JOBLIB_CFG"] = dict()

"""
shuffling method for computing the gcmi_stats_ccd. Use :
   * 'c' : shuffle only the continuous variable
   * 'd' : shuffle only the discret variable
   * 'cd' : shuffle both the continuous and discrete variables (default)
"""
CONFIG["MI_PERM_CCD"] = 'cd'

"""
Several functions can be compiled using Numba. Use this argument to specify if
Numba compilation should be used or not
"""
CONFIG['USE_NUMBA'] = True

"""
MarsAtlas region of interest names
"""
CONFIG['MA_NAMES'] = [
    'L_VCcm', 'L_VCl', 'L_VCs', 'L_Cu', 'L_VCrm', 'L_ITCm', 'L_ITCr', 'L_MTCc',
    'L_STCc', 'L_STCr', 'L_MTCr', 'L_ICC', 'L_IPCv', 'L_IPCd', 'L_SPC',
    'L_SPCm', 'L_PCm', 'L_PCC', 'L_Sv', 'L_Sdl', 'L_Sdm', 'L_Mv', 'L_Mdl',
    'L_Mdm', 'L_PMrv', 'L_PMdl', 'L_PMdm', 'L_PFcdl', 'L_PFcdm', 'L_MCC',
    'L_PFrvl', 'L_Pfrdli', 'L_Pfrdls', 'L_PFrd', 'L_PFrm', 'L_OFCvl', 'L_OFCv',
    'L_OFCvm', 'L_PFCvm', 'L_ACC', 'L_Insula', 'R_VCcm', 'R_VCl', 'R_VCs',
    'R_Cu', 'R_VCrm', 'R_ITCm', 'R_ITCr', 'R_MTCc', 'R_STCc', 'R_STCr',
    'R_MTCr', 'R_ICC', 'R_IPCv', 'R_IPCd', 'R_SPC', 'R_SPCm', 'R_PCm', 'R_PCC',
    'R_Sv', 'R_Sdl', 'R_Sdm', 'R_Mv', 'R_Mdl', 'R_Mdm', 'R_PMrv', 'R_PMdl',
    'R_PMdm', 'R_PFcdl', 'R_PFcdm', 'R_MCC', 'R_PFrvl', 'R_Pfrdli', 'R_Pfrdls',
    'R_PFrd', 'R_PFrm', 'R_OFCvl', 'R_OFCv', 'R_OFCvm', 'R_PFCvm', 'R_ACC',
    'R_Insula', 'L_Thal', 'L_Cd', 'L_Put', 'L_GP', 'L_Hipp', 'L_Amyg', 'L_NAc',
    'R_Thal', 'R_Cd', 'R_Put', 'R_GP', 'R_Hipp', 'R_Amyg', 'R_NAc']


"""
Default sigma for the hat correction when performing the t-test using
MNE-Python
"""
CONFIG['TTEST_MNE_SIGMA'] = 0.001
