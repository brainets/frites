"""Configuration file."""
import mne

# Empty config
CONFIG = dict()

# Supported MNE types
MNE_EPOCHS_TYPE = (mne.Epochs, mne.EpochsArray, mne.epochs.EpochsFIF,
                   mne.time_frequency.EpochsTFR, mne.epochs.BaseEpochs)
CONFIG["MNE_EPOCHS_TYPE"] = MNE_EPOCHS_TYPE

# MarsAtlas ROI names :
MA_NAMES = ['L_VCcm', 'L_VCl', 'L_VCs', 'L_Cu', 'L_VCrm', 'L_ITCm', 'L_ITCr',
            'L_MTCc', 'L_STCc', 'L_STCr', 'L_MTCr', 'L_ICC', 'L_IPCv',
            'L_IPCd', 'L_SPC', 'L_SPCm', 'L_PCm', 'L_PCC', 'L_Sv', 'L_Sdl',
            'L_Sdm', 'L_Mv', 'L_Mdl', 'L_Mdm', 'L_PMrv', 'L_PMdl', 'L_PMdm',
            'L_PFcdl', 'L_PFcdm', 'L_MCC', 'L_PFrvl', 'L_Pfrdli', 'L_Pfrdls',
            'L_PFrd', 'L_PFrm', 'L_OFCvl', 'L_OFCv', 'L_OFCvm', 'L_PFCvm',
            'L_ACC', 'L_Insula', 'R_VCcm', 'R_VCl', 'R_VCs', 'R_Cu', 'R_VCrm',
            'R_ITCm', 'R_ITCr', 'R_MTCc', 'R_STCc', 'R_STCr', 'R_MTCr',
            'R_ICC', 'R_IPCv', 'R_IPCd', 'R_SPC', 'R_SPCm', 'R_PCm', 'R_PCC',
            'R_Sv', 'R_Sdl', 'R_Sdm', 'R_Mv', 'R_Mdl', 'R_Mdm', 'R_PMrv',
            'R_PMdl', 'R_PMdm', 'R_PFcdl', 'R_PFcdm', 'R_MCC', 'R_PFrvl',
            'R_Pfrdli', 'R_Pfrdls', 'R_PFrd', 'R_PFrm', 'R_OFCvl', 'R_OFCv',
            'R_OFCvm', 'R_PFCvm', 'R_ACC', 'R_Insula', 'L_Thal', 'L_Cd',
            'L_Put', 'L_GP', 'L_Hipp', 'L_Amyg', 'L_NAc', 'R_Thal', 'R_Cd',
            'R_Put', 'R_GP', 'R_Hipp', 'R_Amyg', 'R_NAc']
CONFIG["MA_NAMES"] = MA_NAMES

# gcmi configuration
CONFIG["KW_GCMI"] = dict(shape_checking=False, biascorrect=True, demeaned=True,
                         mvaxis=-2, traxis=-1)

# copula name conversion
CONFIG["COPULA_CONV"] = dict(cc='gg', cd='gd', ccd='ggd')

# general joblib config
CONFIG["JOBLIB_CFG"] = dict()

# shuffling method for computing the gcmi_stats_ccd. Use :
#    * 'c' : shuffle only the continuous variable
#    * 'd' : shuffle only the discret variable
#    * 'cd' : shuffle both the continuous and discret variables (default)
CONFIG["MI_PERM_CCD"] = 'cd'
