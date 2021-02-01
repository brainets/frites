"""Container for the electrophysiological data of a single subject."""
import logging
from collections import OrderedDict

import numpy as np
import xarray as xr
import mne

import frites
from frites.io import Attributes, set_log_level
from frites.config import CONFIG

logger = logging.getLogger("frites")


class SubjectEphy(Attributes):
    """Container for the electrophysiological data of a single subject.

    This class can be used to convert the data from different types (e.g
    NumPy, MNE-Python, Xarray) into a single format (xarray.DataArray).

    Parameters
    ----------
    x : array_like
        The electrophysiological data of a single subject. Several data types
        are supported :

            * 3d NumPy array of shape (n_epochs, n_channels, n_times)
            * 4d NumPy array of shape (n_epochs, n_channels, n_freqs, n_times)
            * 4d NumPy array of shape (n_epochs, n_channels, mv, n_times)
              where 'mv' refers to an axis to consider as multi-variate
            * mne.Epochs or mne.EpochsArray
            * mne.EpochsTFR (i.e. non-averaged power)
            * xarray.DataArray. In that case `y`, `z`, `roi` and `times` inputs
              can be strings that refer to the coordinate name to use in the
              DataArray

    y, z : list, sting | None
        Task-related variables (e.g discret stimulus, learning rate etc.) The
        y and z variables must be vectors of shapes (n_epochs,). The MI is then
        going to be computed between the data (x) and those task related
        variables. The type of MI depends on the type of this two variables :

            * y=continuous, z=None : I(x; y) and mi_type should be 'cc'
            * y=discrete, z=None : I(x; y) and mi_type should be 'cd'
            * y=continuous, z=discrete : I(x; y | z) and mi_type should
              be 'ccd'

        Note that if y (or z) are multi-dimensional discrete variables, the
        categories inside are going to be automatically remapped to a single
        vector. Several input types are supported :

            * A NumPy array of shape (n_epochs,)
            * If x is a DataArray, the dimension name to use to infer the
              task-related variable

    roi : list | None
        Anatomical informations of each channel / electrodes. Several input
        types are supported :

            * A NumPy array of shape (n_channels). If None, unique ROI are
              defined ('roi_0', 'roi_1' etc.)
            * If None, and if input type is coming from MNE, it's using the
              `ch_names` (info['ch_names'])
            * If x is a DataArray, the dimension name to use to infer the
              anatomical informations

    sub_roi : list, string | None
        Additional anatomical informations about the data of each subject. This
        argument can be used to estimate the mutual-information at the single
        contact level (or MEG sensor for exemple). As an example, the `roi`
        input could refer to the parcel name (e.g 'Insula', 'vmPFC' etc.) while
        the `sub_roi` input can refer to the contact / sensor name
        ('x1', 'x2' etc.). Several input types are supported :

            * A NumPy array of length (n_channels)
            * The dimension name to use to infer the additional anatomical
              information

    times : array_like | None
        The time vector to use. Several types are supported :

            * A NumPy array of length (n_times,)
            * If MNE object are passed, the time vector is automatically
              inferred from it
            * If x is a DataArray, the dimension name to use to infer to the
              time vector

    multivariate : bool | False
        If 4d input is provided, this parameter specifies whether this axis
        should be considered as multi-variate (True) or uni-varariate (False)
    name : string | None
        Subject name
    sfreq : float | None
        The sampling frequency. If None, it could be inferred if the time
        vector is provided or if the input x is an MNE object

    Returns
    -------
    da : xarray.DataArray
        A formatted DataArray representing the subject with all the
        task-related variables inside
    """

    def __new__(self, x, y=None, z=None, roi=None, sub_roi=None, times=None,
                multivariate=False, name=None, attrs=None, sfreq=None,
                verbose=None):
        """Init."""
        set_log_level(verbose)
        attrs = Attributes(attrs=attrs)
        _supp_dim = []

        # ========================= Inputs conversion =========================

        # ____________________________ extraction _____________________________
        if isinstance(x, xr.DataArray):  # xr -> xr
            # get data, name and attributes
            attrs.update(x.attrs)
            name = x.name if name is None else name
            data = x.data
            # get y / z regressors
            y = x[y].data if isinstance(y, str) else y
            z = x[z].data if isinstance(z, str) else z
            # get spatial informations (roi & sub_roi)
            roi = x[roi].data if isinstance(roi, str) else roi
            sub_roi = x[sub_roi].data if isinstance(sub_roi, str) else sub_roi
            # build 4d (possibly multivariate) coordinate
            if x.ndim == 4:
                if multivariate:
                    _supp_dim = ('mv', np.full((x.shape[2]), np.nan))
                else:
                    _supp_dim = (x.dims[2], x[x.dims[2]].data)
            # get the temporal vector
            times = x[times].data if isinstance(times, str) else times

        if 'mne' in str(type(x)):       # mne -> xr
            times = x.times if times is None else times
            roi = x.info['ch_names'] if roi is None else roi
            sfreq = x.info['sfreq'] if sfreq is None else sfreq
            if isinstance(x, CONFIG["MNE_EPOCHS_TYPE"]):
                data = x.get_data()
            elif isinstance(x, CONFIG["MNE_EPOCHSTFR_TYPE"]):
                data = x.data
                if multivariate:
                    _supp_dim = ('mv', np.full((data.shape[2]), np.nan))
                else:
                    _supp_dim = ('freqs', x.freqs)

        if isinstance(x, np.ndarray):    # numpy -> xr
            data = x
            if data.ndim == 4:
                if multivariate:
                    _supp_dim = ('mv', np.full((data.shape[2]), np.nan))
                else:
                    _supp_dim = ('supp', np.arange(data.shape[2]))

        assert data.ndim <= 4, "Data up to 4-dimensions are supported"

        # __________________________ sampling rate ____________________________
        # infer the sampling frequency (if needed)
        if sfreq is None:
            if times is not None:
                sfreq = 1. / (times[1] - times[0])
            else:
                logger.warning("Impossible to infer the sampling frequency. "
                               "You should consider providing a time vector")
                sfreq = 1.
        sfreq = float(sfreq)

        # _______________________________ y / z _______________________________

        # infer dtypes
        y_dtype = self._infer_dtypes(y, 'y')
        z_dtype = self._infer_dtypes(z, 'z')

        # ============================= DataArray =============================

        # ___________________________ dims & coords ___________________________

        dims, coords = [], OrderedDict()
        n_trials, n_roi, n_times = np.array(list(data.shape))[[0, 1, -1]]

        # don't break if empty time vector or missing roi
        if times is None:
            logger.warning("No time vector. A default one is created")
            times = np.arange(n_times) / sfreq
        if roi is None:
            logger.warning("No regions of interest are provided (roi). Default"
                           " ones are created")
            roi = [f"roi_{k}" for k in range(n_roi)]

        # build trials (potentially) multi-coordinates
        coords['trials'] = ('trials', np.arange(n_trials))
        if (y is not None) and (len(y) == n_trials):
            coords['y'] = ('trials', y)
        if (z is not None) and (len(z) == n_trials):
            coords['z'] = ('trials', z)
        if name is not None:
            coords['subject'] = ('trials', [name] * n_trials)
        dims += ['trials']
        # build space (potentially) multi-coordinates
        coords['space'] = ('space', np.arange(n_roi))
        if (roi is not None) and (len(roi) == n_roi):
            coords['roi'] = ('space', roi)
        if (sub_roi is not None) and (len(sub_roi) == n_roi):
            coords['sub_roi'] = ('space', sub_roi)
        dims += ['space']
        if _supp_dim:
            coords[_supp_dim[0]] = _supp_dim[1]
            dims += [_supp_dim[0]]
        # build temporal coordinate
        if (times is not None) and (len(times) == n_times):
            coords['times'] = ('times', times)
        dims += ['times']

        # _______________________ additional attributes _______________________
        attrs.update({
            '__version__': frites.__version__,
            'modality': "electrophysiology",
            'dtype': 'SubjectEphy',
            'y_dtype': y_dtype,
            'z_dtype': z_dtype,
            'sfreq': sfreq,
        })

        # _____________________________ dataarray _____________________________
        # for a given reason, DataArray are not easy to subclass (see #706,
        # #728, #3980). Therefore, for the moment, it's just easier to simply
        # return a dataarray
        da = xr.DataArray(data, dims=dims, coords=coords, name=name,
                          attrs=attrs)

        return da

    @staticmethod
    def _infer_dtypes(var, var_name):
        """Check that the dtypes of list of variables is consistent."""
        # evacuate none
        if var is None:
            return 'none'
        dtype = np.asarray(var).dtype
        print(dtype)
        if dtype in CONFIG['INT_DTYPE']:
            return 'int'
        elif dtype in CONFIG['FLOAT_DTYPE']:
            return 'float'
        else:
            raise TypeError(f"{dtype} type for input {var_name} is unknown")
