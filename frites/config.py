"""Configuration file."""
import mne

# Empty config
CONFIG = dict()

# Supported MNE types
MNE_EPOCHS_TYPE = (mne.Epochs, mne.EpochsArray, mne.epochs.EpochsFIF,
                   mne.time_frequency.EpochsTFR, mne.epochs.BaseEpochs)
CONFIG["MNE_EPOCHS_TYPE"] = MNE_EPOCHS_TYPE

# gcmi configuration
CONFIG["KW_GCMI"] = dict(shape_checking=False, biascorrect=True, demeaned=True,
                         mvaxis=-2, traxis=-1)

# copula name conversion
CONFIG["COPULA_CONV"] = dict(cc='gg', cd='gd', ccd='ggd')

# general joblib config
CONFIG["JOBLIB_CFG"] = dict()

"""
shuffling method for computing the gcmi_stats_ccd. Use :
   * 'c' : shuffle only the continuous variable
   * 'd' : shuffle only the discret variable
   * 'cd' : shuffle both the continuous and discret variables (default)
"""
CONFIG["MI_PERM_CCD"] = 'cd'

"""
Several functions can be compiled using Numba. Use this argument to specify if
Numba compilation should be used or not
"""
CONFIG['USE_NUMBA'] = True
