"""Dataset representation of electrophysiological data."""
import logging

import numpy as np

import frites
from frites.config import CONFIG
from frites.core import copnorm_cat_nd, copnorm_nd
from frites.io import set_log_level

logger = logging.getLogger("frites")


class DatasetEphy(object):
    """Dataset of electrophysiological data coming from several subjects.

    This class is used to represent the neurophysiological data coming from
    multiple subjects. Then, the created object can be used to compute the
    mutual information (MI) and perform statistics on it.

    Parameters
    ----------
    x : list
        List of length (n_subjects,). Each element of the list should either be
        an array of shape (n_epochs, n_channels, n_times), mne.Epochs,
        mne.EpochsArray, mne.EpochsTFR (i.e. non-averaged power).
    roi : list
        List of length (n_subjects,) where each element is an array of shape
        (n_channels,) describing the ROI name of each channel.
    y, z : list
        List of length (n_subjects,) of continuous or discrete variables. Each
        element of the list should be an array of shape (n_epochs,) that is
        then going to be used to compute the MI. The type of MI depends on the
        type of this two variables :

            * y=continuous, z=None : I(x; y) and mi_type should be 'cc'
            * y=discrete, z=None : I(x; y) and mi_type should be 'cd'
            * y=continuous, z=discrete : I(x; y | z) and mi_type should
              be 'ccd'

        Note that if y (or z) are multi-dimensional discrete variables, the
        categories inside are going to be automatically remapped to a single
        vector.
    times : array_like | None
        The time vector to use. If the data are defined using MNE-Python, the
        time vector is directly inferred from those files.
    nb_min_suj : int | None
        The minimum number of subjects per roi. Roi with n_suj < nb_min_suj
        are going to be skipped. Use None to skip this parameter
    """

    def __init__(self, x, y, roi, z=None, times=None, nb_min_suj=None,
                 verbose=None):
        """Init."""
        set_log_level(verbose)
        # ---------------------------------------------------------------------
        # check input

        assert all([isinstance(k, (list, tuple)) for k in (x, y)])
        assert len(x) == len(y) == len(roi), (
            "the data (x), condition variable (y) and roi must all be lists "
            "with a length of `n_subjects`")
        roi = [np.asarray(k) for k in roi]

        # data related
        self.nb_min_suj = nb_min_suj
        self.n_subjects = len(x)
        self.times = times
        self.roi = roi
        # unique roi list
        merged_roi = np.r_[tuple(self.roi)]
        _, u_idx = np.unique(merged_roi, return_index=True)
        self.roi_names = merged_roi[np.sort(u_idx)]
        self.n_roi = len(self.roi_names)

        # internals
        self.modality = "electrophysiological"
        self._copnormed = False
        self._groupedby = "subject"
        self.__version__ = frites.__version__

        logger.info(f"Dataset composed of {self.n_subjects} subjects. At least"
                    f" {self.nb_min_suj} subjects per roi are required")

        # ---------------------------------------------------------------------
        # check the types of y (and z)
        self._y_dtype = self._check_dtypes(y, 'y')
        self._z_dtype = self._check_dtypes(z, 'z')
        if (self._y_dtype == 'float') and (self._z_dtype == 'none'):
            self._mi_type = 'cc'
        elif (self._y_dtype == 'int') and (self._z_dtype == 'none'):
            self._mi_type = 'cd'
        elif (self._y_dtype == 'float') and (self._z_dtype == 'int'):
            self._mi_type = 'ccd'
        else:
            raise TypeError(f"Types of y ({self._y_dtype}) and z ("
                            f"{self._z_dtype}) doesn't allow to then compute "
                            "mi on it")
        logger.info(f"    Allowed mi_type={self._mi_type} (y.dtype="
                    f"{self._y_dtype}; z.dtype={self._z_dtype})")
        # (optionnal) multi-conditions conversion
        if self._y_dtype == 'int':
            y = self._multicond_conversion(y, 'y', verbose)
        if self._z_dtype == 'int':
            z = self._multicond_conversion(z, 'z', verbose)

        # ---------------------------------------------------------------------
        # load the data of each subject

        logger.info("    Load the data of each subject")
        self._x = [self._load_single_suj_ephy(x[k]) for k in range(
            self.n_subjects)]
        self._y = [np.asarray(k) for k in y]
        self._z = z

        # check the time vector
        _x_times = np.unique([self._x[k].shape[1] for k in range(
            self.n_subjects)])
        assert _x_times.size == 1, ("Inconsistent number of time points across"
                                    " subjects")
        self.n_times = self._x[0].shape[1]
        if not isinstance(self.times, np.ndarray):
            logger.warning("No time vector found. A default will be used "
                           "instead. You should use the `times` input instead")
            self.times = np.arange(self.n_times)
        self.sfreq = 1. / (self.times[1] - self.times[0])

        # check consistency between x and y
        _const = [self._x[k].shape == (len(roi[k]), self.n_times,
                                       len(self._y[k])) for k in range(
            self.n_subjects)]
        assert all(_const), "Inconsistent shape between `x`, `y` and `roi`"

    ###########################################################################
    # INTERNALS
    ###########################################################################

    def __repr__(self):
        """String representation."""
        sep = '-' * 79
        rep = (f"{sep}\n"
               f"number of (subjects, roi, time points) : "
               f"{self.n_subjects, self.n_roi, self.n_times}\n"
               f"minimum number of subject per roi: {self.nb_min_suj}\n"
               f"modality : {self.modality}\n"
               f"data grouped by : {self._groupedby}\n"
               f"copnormed : {self._copnormed}\n"
               f"version : {self.__version__}\n"
               f"{sep}")
        return rep

    def __len__(self):
        """Get the number of subjects."""
        return self.n_subjects

    def __getitem__(self, arg):
        """Slice the dataset."""
        assert self._groupedby is "subject", ("Slicing only work when data is "
                                              "grouped by 'subjects'")
        if isinstance(arg, slice):
            arg = (arg, slice(None, None, None))
        if len(arg) == 1:
            arg = (arg[0], slice(None, None, None))
        assert len(arg) in [1, 2]
        sl_time, sl_roi = arg
        if isinstance(sl_roi, str):
            sl_roi = [sl_roi]
        # time slicing
        slt_start = self.__slice_float(sl_time.start, self.times)
        slt_stop = self.__slice_float(sl_time.stop, self.times)
        sl_time = slice(slt_start, slt_stop, sl_time.step)
        self._x = [k[:, sl_time, :] for k in self._x]
        self.times = self.times[sl_time]
        self.n_times = self._x[0].shape[1]
        # roi slicing
        if isinstance(sl_roi, (tuple, list, np.ndarray)):
            for n_s in range(self.n_subjects):
                s_roi = np.asarray(self.roi[n_s])
                is_roi = np.zeros((len(s_roi), len(sl_roi)))
                for n_r, r in enumerate(sl_roi):
                    is_roi[:, n_r] = s_roi == r
                is_roi = is_roi.any(axis=1)
                self._x[n_s] = self._x[n_s][is_roi, ...]
                self.roi[n_s] = s_roi[is_roi]
            # unique roi list
            merged_roi = np.r_[tuple(self.roi)]
            _, u_idx = np.unique(merged_roi, return_index=True)
            self.roi_names = merged_roi[np.sort(u_idx)]
            self.n_roi = len(self.roi_names)

        return self

    @staticmethod
    def __slice_float(ref, vec):
        """Find closest index in a floating vector."""
        if isinstance(ref, (int, float)):
            ref = np.abs(vec - ref).argmin()
        else:
            ref = ref
        return ref

    @staticmethod
    def _check_dtypes(var, var_name):
        """Check that the dtypes of list of variables is consistent."""
        # evacuate none
        if var is None:
            return 'none'
        assert isinstance(var, (list, tuple))
        # get dtypes
        dtypes = [k.dtype == var[0].dtype for k in var]
        if not all(dtypes):
            raise TypeError(f"Arrays in {var_name} input doesn't have the same"
                            " types.")
        dtype = var[0].dtype
        if dtype in CONFIG['INT_DTYPE']:
            return 'int'
        elif dtype in CONFIG['FLOAT_DTYPE']:
            return 'float'
        else:
            raise TypeError(f"{dtype} type for input {var_name} is unknown")

    @staticmethod
    def _multicond_conversion(x, var_name, verbose):
        """Convert a discret vector that contains multiple conditions."""
        if not isinstance(x, (list, tuple, np.ndarray)):
            return x
        x = [np.asarray(k) for k in x]
        # get if all variables are integers and multicolumns else skip it
        is_int = all([k.dtype in CONFIG['INT_DTYPE'] for k in x])
        is_ndim = all([k.ndim > 1 for k in x])
        if not is_int or not is_ndim:
            return x
        # test that all dimensions are equals
        same_dim = all([k.ndim == x[0].ndim for k in x])
        if not same_dim:
            assert ValueError(f"Every array in the `{var_name}` input should "
                              "have the same number of dimensions")
        # otherwise find all possible pairs
        x_all = np.concatenate(x, axis=0)
        idx = np.unique(x_all, axis=0, return_index=True)[1]
        u_cat = x_all[sorted(idx), :]
        # show to the user the new categories
        user = []
        for n_c, cat in enumerate(u_cat):
            user += [f"{n_c}: [{', '.join([str(c) for c in cat])}]"]
        logger.info(f"    The `{var_name}` input contains multiple conditions "
                    f"that have been remapped to : {'; '.join(user)}")
        # loop over subjects
        x_new = []
        for k in range(len(x)):
            x_cat = np.full((x[k].shape[0],), -1, dtype=int)
            for n_c, cat in enumerate(u_cat):
                x_cat[np.equal(x[k], cat.reshape(1, -1)).all(1)] = n_c
            assert x_cat.min() > -1, "Not all values have replaced"
            x_new += [x_cat]

        return x_new

    ###########################################################################
    # METHODS
    ###########################################################################

    def _load_single_suj_ephy(self, x_suj):
        """Load the data of a single subject.

        This method returns an array of shape (n_roi, n_times, n_trials).
        """
        # Check inputs
        if isinstance(x_suj, CONFIG["MNE_EPOCHS_TYPE"]):
            self.times = x_suj.times
            data = x_suj.get_data()
        elif isinstance(x_suj, np.ndarray) and (x_suj.ndim == 3):
            data = x_suj
        else:
            raise TypeError(f"data type {type(x_suj)} not supported")
        assert isinstance(data, np.ndarray)

        # Handle multi-dimentional arrays
        if data.ndim == 4:  # TF : (n_trials, n_channels, n_freqs, n_pts)
            if data.shape[2] == 1:
                data = data[..., 0, :]
            else:
                data = data.mean(2)
                logger.warning("Multiple frequencies detected. Take the mean "
                               "across frequencies")
        assert (data.ndim == 3), ("data should be a 3D array of shape "
                                  "(n_trials, n_channels, n_pts)")

        # mne data are (n_epochs, n_channels, n_times). Here, we move the trial
        # axis to the end
        return np.moveaxis(data, 0, -1)

    def groupby(self, groupby="roi"):
        """Reorganize the data inplace.

        Parameters
        ----------
        groupby : {"roi", "subject"}
            Group data either by subject or by roi
        """
        assert groupby in ["subject", "roi"]
        if groupby == self._groupedby:
            logger.warning("Grouping ignored because already grouped by "
                           f"{self._groupedby}")
            return
        logger.info(f"    Group data by {groupby}")

        if groupby == "roi":  # -----------------------------------------------
            # be sure that y is at least (n_epochs, 1)
            for k in range(len(self._y)):
                if self._y[k].ndim == 1:
                    self._y[k] = self._y[k][:, np.newaxis]
            n_cols_y = self._y[0].shape[1]
            # then merge (y, z)
            if isinstance(self._z, list):  # CCD
                assert all([k.shape[0] == i.shape[0] for k, i in zip(
                    self._y, self._z)]), ("y and z must have the same number "
                                          "of epochs")
                yz = [np.c_[k, i] for k, i in zip(self._y, self._z)]
            else:
                yz = self._y
            # group by roi
            roi, x_roi, yz_roi, suj_roi, suj_roi_u = [], [], [], [], []
            roi_ignored = []
            for r in self.roi_names:
                # loop over subjects to find if roi is present. If not, discard
                _x, _yz, _suj, _suj_u = [], [], [], []
                for n_s, data in enumerate(self._x):
                    # skip missing roi
                    if r not in self.roi[n_s]:
                        continue  # noqa
                    # sEEG data can have multiple sites inside a specific roi
                    # so we need to identify thos sites
                    idx = self.roi[n_s] == r
                    __x = np.array(data[idx, ...]).squeeze()
                    __yz = yz[n_s]
                    # in case there's multiple sites in this roi, we reshape
                    # as if the data were coming from a single site, hence
                    # increasing the number of trials
                    n_sites = idx.sum()
                    if n_sites != 1:
                        ___x, ___yz = [], []
                        for _ne in range(n_sites):
                            ___x += [__x[_ne, ...]]
                            ___yz += [__yz]
                        __x = np.concatenate(___x, axis=1)
                        __yz = np.concatenate(___yz, axis=0)
                        del ___x, ___yz
                    # at this point the data are (n_times, n_epochs)
                    _x += [__x]
                    _yz += [__yz]
                    _suj += [n_s] * len(__yz)
                    _suj_u += [n_s]
                # test if the minimum number of unique subject is met inside
                # the roi
                u_suj = len(np.unique(_suj))
                if u_suj < self.nb_min_suj:
                    roi_ignored += [r]
                    continue
                # concatenate across the trial axis
                _x = np.concatenate(_x, axis=1)
                _yz = np.r_[tuple(_yz)]
                _suj = np.array(_suj)
                # keep latest version
                x_roi += [_x[:, np.newaxis, :]]
                yz_roi += [_yz]
                suj_roi += [_suj]
                suj_roi_u += [np.array(_suj_u)]
                roi += [r]
            # test if the data are not empty
            assert len(x_roi), ("Empty dataset probably because `nb_min_suj` "
                                "is too high for your dataset")
            # warning message if there's ignored ROIs
            if len(roi_ignored):
                logger.warning("The following roi have been ignored because of"
                               f" a number of subjects bellow "
                               f"{self.nb_min_suj} : {', '.join(roi_ignored)}")
            # update variables
            self._x = x_roi
            if not isinstance(self._z, list):
                self._y = yz_roi
            else:
                self._y = [k[:, 0:n_cols_y] for k in yz_roi]
                self._z = [k[:, n_cols_y:].astype(int) for k in yz_roi]
            self.suj_roi = suj_roi
            self.suj_roi_u = suj_roi_u
            self.roi_names = roi
            self.n_roi = len(roi)
        elif groupby == "subject":  # -----------------------------------------
            raise NotImplementedError("TODO FOR FIT + TRANSFERT ENTROPY")

        self._groupedby = groupby

    def copnorm(self, mi_type='cc', gcrn_per_suj=True):
        """Apply the Gaussian-Copula rank normalization.

        The copnorm is only applied to continuous variables.

        Parameters
        ----------
        mi_type : {'cc', 'cd', 'ccd'}
            The copnorm depends on the mutual-information type that is going to
            be performed. Choose either 'cc' (continuous / continuous), 'cd'
            (continuous / discret) or 'ccd' (continuous / continuous / discret)
        gcrn_per_suj : bool | True
            Apply the Gaussian-rank normalization either per subject (True)
            or across subjects (False).
        """
        assert mi_type in ['cc', 'cd', 'ccd']
        # do not enable to copnorm two times
        if isinstance(self._copnormed, str):
            logger.warning("Data already copnormed. Copnorm ignored")
            return None
        logger.info(f"    Apply copnorm (per subject={gcrn_per_suj}; "
                    f"mi_type={mi_type})")
        # copnorm applied differently how data have been organized
        if self._groupedby == "roi":
            if gcrn_per_suj:  # per subject
                logger.debug("copnorm applied per subjects")
                self._x = [copnorm_cat_nd(k, i, axis=-1) for k, i in zip(
                    self._x, self.suj_roi)]
                if mi_type in ['cc', 'ccd']:
                    self._y = [copnorm_cat_nd(k, i, axis=0) for k, i in zip(
                        self._y, self.suj_roi)]
            else:             # subject-wise
                logger.debug("copnorm applied across subjects")
                self._x = [copnorm_nd(k, axis=-1) for k in self._x]
                if mi_type in ['cc', 'ccd']:
                    self._y = [copnorm_nd(k, axis=0) for k in self._y]
        elif self._groupedby == "subject":
            raise NotImplementedError("FUTURE WORK")

        self._copnormed = f"{int(gcrn_per_suj)}-{mi_type}"

    def savgol_filter(self, h_freq, verbose=None):
        """Filter the data using Savitzky-Golay polynomial method.

        This method is an adaptation of the mne-python one.

        Parameters
        ----------
        h_freq : float
            Approximate high cut-off frequency in Hz. Note that this is not an
            exact cutoff, since Savitzky-Golay filtering is done using
            polynomial fits instead of FIR/IIR filtering. This parameter is
            thus used to determine the length of the window over which a
            5th-order polynomial smoothing is used.

        Returns
        -------
        inst : instance of DatasetEphy
            The object with the filtering applied.

        Notes
        -----
        For Savitzky-Golay low-pass approximation, see:
            https://gist.github.com/larsoner/bbac101d50176611136b
        """
        set_log_level(verbose)
        assert self._groupedby is "subject", ("Slicing only work when data is "
                                              "grouped by 'subjects'")
        from scipy.signal import savgol_filter
        h_freq = float(h_freq)
        if h_freq >= self.sfreq / 2.:
            raise ValueError('h_freq must be less than half the sample rate')

        # savitzky-golay filtering
        window_length = (int(np.round(self.sfreq / h_freq)) // 2) * 2 + 1
        logger.info(f'    Using savgol length {window_length}')
        for k in range(len(self._x)):
            self._x[k] = savgol_filter(self._x[k], axis=1, polyorder=5,
                                      window_length=window_length)
        return self

    def resample(self, sfreq, npad='auto', window='boxcar', n_jobs=1,
                 pad='edge', verbose=None):
        """Resample data.

        This method is an adaptation of the mne-python one.

        Parameters
        ----------
        sfreq : float
            New sample rate to use.
        npad : int | str
            Amount to pad the start and end of the data. Can also be “auto” to
            use a padding that will result in a power-of-two size (can be much
            faster).
        window : str | tuple
            Frequency-domain window to use in resampling. See
            scipy.signal.resample().
        pad : str | 'edge'
            The type of padding to use. Supports all numpy.pad() mode options.
            Can also be “reflect_limited”, which pads with a reflected version
            of each vector mirrored on the first and last values of the vector,
            followed by zeros. Only used for method='fir'. The default is
            'edge', which pads with the edge values of each vector.

        Returns
        -------
        inst : instance of DatasetEphy
            The object with the filtering applied.

        Notes
        -----
        For some data, it may be more accurate to use npad=0 to reduce
        artifacts. This is dataset dependent -- check your data!
        """
        set_log_level(verbose)
        assert self._groupedby is "subject", ("Slicing only work when data is "
                                              "grouped by 'subjects'")
        from mne.filter import resample
        sfreq = float(sfreq)
        o_sfreq = self.sfreq
        logger.info(f"    Resample to the frequency {sfreq}Hz")
        for k in range(len(self._x)):
            _x = np.transpose(self._x[k], axes=(0, 2, 1))
            _x = resample(_x, sfreq, o_sfreq, npad, window=window,
                          n_jobs=n_jobs, pad=pad)
            self._x[k] = np.transpose(_x, axes=(0, 2, 1))
        self.sfreq = float(sfreq)

        self.times = (np.arange(self._x[0].shape[1], dtype=np.float) /
                      sfreq + self.times[0])
        self.n_times = len(self.times)

        return self

    def save(self):
        """Save the dataset."""
        raise NotImplementedError("FUTURE WORK")

    ###########################################################################
    # PROPERTIES
    ###########################################################################

    @property
    def x(self):
        """Get the data."""
        return self._x

    @property
    def y(self):
        """Get the y value."""
        return self._y

    @property
    def z(self):
        """Get the z value."""
        if self._z is None:
            return [None] * len(self._x)
        else:
            return self._z

    @property
    def nb_min_suj(self):
        """Get the minimum number of subjects needed per roi."""
        return self._nb_min_suj

    @nb_min_suj.setter
    def nb_min_suj(self, value):
        """Set nb_min_suj value."""
        self._nb_min_suj = -np.inf if not isinstance(value, int) else value

    @property
    def shape(self):
        """Get the shape of the x, y and z (if defined)."""
        _ssh = [0, 1, None, -1]
        _xsh = [str(self._x[k].shape) if isinstance(
            k, int) else "..." for k in _ssh]
        _ysh = [str(self._y[k].shape) if isinstance(
            k, int) else "..." for k in _ssh]
        if isinstance(self._z, list):  # Oh my zsh
            _zsh = [str(self._z[k].shape) if isinstance(
                k, int) else "..." for k in _ssh]
            _zpr = (f"z ({len(self._z)} x {self._z[0].dtype}) : "
                    f"{', '.join(_zsh)}")
        else:
            _zpr = f"z : {None}"

        shape = (
            f"{'-' * 79}\n"
            f"x ({len(self._x)} x {self._x[0].dtype}) : {', '.join(_xsh)}\n"
            f"y ({len(self._y)} x {self._y[0].dtype}) : {', '.join(_ysh)}\n"
            f"{_zpr}\n"
            f"{'-' * 79}")
        return shape

if __name__ == '__main__':
    import numpy as np

    x = [np.random.rand(10, 1, 20) for k in range(4)]
    y = [np.random.rand(10) for k in range(4)]
    # y = [np.random.randint(0, 1, (10,)) for k in range(4)]
    z = [np.random.randint(0, 2, (10, 2)) for k in range(4)]
    roi = [np.array(['VCcm'])] * 4
    times = np.linspace(-1, 1, 20)

    DatasetEphy(x, y, roi, times=times, z=z)