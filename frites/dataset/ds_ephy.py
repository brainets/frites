"""Dataset representation of electrophysiological data."""
import logging

import numpy as np

import frites
from frites.config import CONFIG
from frites.core import copnorm_cat_nd, copnorm_nd

logger = logging.getLogger("frites")


class DatasetEphy(object):
    """Dataset of electrophysiological data coming from several subjects.

    This class is used to represent the neurophysiological data coming from
    multiple subjects. Then, the created object can be used to compute the
    mutual information and perform statistics on it.

    Parameters
    ----------
    x : list
        List of length (n_subjects,). Each element of the list should either be
        an array of shape (n_epochs, n_channels, n_times), mne.Epochs,
        mne.EpochsArray, mne.EpochsTFR (i.e. non-averaged power).
    roi : list
        List of arrays of shape (n_channels,) describing the ROI name of each
        channel.
    y, z : list
        List of length (n_subjects,) of continuous or discret variables. Each
        element of the list should be an array of shape (n_epochs,) describing
        the continuous variable
    times : array_like | None
        The time vector to use. If the data are defined using MNE-Python, the
        time vector is directly infered from thos files.
    nb_min_suj : int | 10
        The minimum number of subjects per roi. Roi with n_suj < nb_min_suj
        are going to be skipped. Use None to skip this parameter
    """

    def __init__(self, x, y, roi, z=None, times=None, nb_min_suj=None):
        """Init."""
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

        logger.info(f"Creation of a dataset composed with {self.n_subjects} "
                    f"subjects. A minimum of {self.nb_min_suj} per roi is "
                    "required")

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
                           "instead")
            self.times = np.arange(self.n_times)

        # check consistency between x and y
        _const = [self._x[k].shape == (len(roi[k]), self.n_times,
                                       len(self._y[k])) for k in range(
            self.n_subjects)]
        assert all(_const), "Inconsistent shape between x, y and roi"

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

    def __getitem__(self, idx):
        return self._x[idx]

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
            # first, start by merging (y, z)
            if isinstance(self._z, list) and all([
                k.shape == i.shape for k, i in zip(self._y, self._z)]):
                self._y = [np.c_[k, i] for k, i in zip(self._y, self._z)]
            # group by roi
            roi, x_roi, y_roi, suj_roi, suj_roi_u = [], [], [], [], []
            for r in self.roi_names:
                # loop over subjects to find if roi is present. If not, discard
                _x, _y, _suj, _suj_u = [], [], [], []
                for n_s, data in enumerate(self._x):
                    # skip missing roi
                    if r not in self.roi[n_s]:
                        continue  # noqa
                    # sEEG data can have multiple sites inside a specific roi
                    # so we need to identify thos sites
                    idx = self.roi[n_s] == r
                    __x = np.array(data[idx, ...]).squeeze()
                    __y = self._y[n_s]
                    # in case there's multiple sites in this roi, we reshape
                    # as if the data were coming from a single site, hence
                    # increasing the number of trials
                    n_sites = idx.sum()
                    if n_sites != 1:
                        __x = np.moveaxis(__x, 0, -1).reshape(self.n_times, -1)
                        if __y.ndim == 1: __y = __y[:, np.newaxis]  # noqa
                        __y = np.tile(__y, (n_sites, 1)).squeeze()
                    # at this point the data are (n_times, n_epochs)
                    _x += [__x]
                    _y += [__y]
                    _suj += [n_s] * len(__y)
                    _suj_u += [n_s]
                # test if the minimum number of unique subject is met inside
                # the roi
                u_suj = len(np.unique(_suj))
                if u_suj < self.nb_min_suj:
                    logger.warning(f"ROI {r} ignored because there's only "
                                   f"{u_suj} inside")
                    continue
                # concatenate across the trial axis
                _x = np.concatenate(_x, axis=1)
                _y = np.r_[tuple(_y)]
                _suj = np.array(_suj)
                # keep latest version
                x_roi += [_x[:, np.newaxis, :]]
                y_roi += [_y]
                suj_roi += [_suj]
                suj_roi_u += [np.array(_suj_u)]
                roi += [r]
            # update variables
            self._x = x_roi
            if self._y[0].ndim == 1:
                self._y = y_roi
            else:
                self._y = [k[:, 0] for k in y_roi]
                self._z = [k[:, 1:].astype(int) for k in y_roi]
            self.suj_roi = suj_roi
            self.suj_roi_u = suj_roi_u
            self.roi_names = roi
        elif groupby == "subject":  # -----------------------------------------
            raise NotImplementedError("TODO FOR FIT + TRANSFERT ENTROPY")

        self._groupedby = groupby

    def copnorm(self, condition='cc', inference='rfx'):
        """Apply the Gaussian-Copula rank normalization.

        The copnorm is only applied to continuous variables.

        Parameters
        ----------
        condition : {'cc', 'cd', 'ccd'}
            The copnorm depends on the mutual-information type that is going to
            be performed. Choose either 'cc' (continuous / continuous), 'cd'
            (continuous / discret) or 'ccd' (continuous / continuous / discret)
        inference : {'rfx', 'ffx'}
            The copnorm also depends on the inference type. Choose either 'ffx'
            (fixed effect) to apply the copnorm across subjects or 'rfx' (
            random effect) to apply the copnorm per subject.
        """
        assert condition in ['cc', 'cd', 'ccd']
        assert inference in ['rfx', 'ffx']
        # do not enable to copnorm two times
        if isinstance(self._copnormed, str):
            logger.warning("Data already copnormed. Copnorm ignored")
            return None
        logger.info(f"    Apply copnorm (condition={condition}; "
                    f"inference={inference})")
        # copnorm applied differently how data have been organized
        if self._groupedby == "roi":
            if inference == 'ffx':
                # for the fixed effect (ffx) the copnorm is applied across
                # subjects across all space and time
                logger.debug("copnorm applied across subjects")
                self._x = [copnorm_nd(k, axis=-1) for k in self._x]
                if condition in ['cc', 'ccd']:
                    self._y = [copnorm_nd(k, axis=0) for k in self._y]
            elif inference == 'rfx':
                # for the random effect (rfx) the copnorm is applied per
                # subject across space and time
                logger.debug("copnorm applied per subjects")
                self._x = [copnorm_cat_nd(k, i, axis=-1) for k, i in zip(
                    self._x, self.suj_roi)]
                if condition in ['cc', 'ccd']:
                    self._y = [copnorm_cat_nd(k, i, axis=0) for k, i in zip(
                        self._y, self.suj_roi)]
        elif self._groupedby == "subject":
            raise NotImplementedError("FUTURE WORK")

        self._copnormed = f"{condition} - {inference}"

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

        shape = (f"{'-' * 79}\n"
            f"x ({len(self._x)} x {self._x[0].dtype}) : {', '.join(_xsh)}\n"
            f"y ({len(self._y)} x {self._y[0].dtype}) : {', '.join(_ysh)}\n"
            f"{_zpr}\n"
            f"{'-' * 79}")
        return shape



if __name__ == '__main__':
    from frites.simulations import sim_multi_suj_ephy

    modality = 'intra'
    n_subjects = 4
    n_epochs = 10
    n_times = 100
    n_roi = 5
    n_sites_per_roi = 3
    as_mne = False
    x, roi, time = sim_multi_suj_ephy(n_subjects=n_subjects, n_epochs=n_epochs,
                                      n_times=n_times, n_roi=n_roi,
                                      n_sites_per_roi=n_sites_per_roi,
                                      as_mne=as_mne, modality=modality,
                                      random_state=1)
    if as_mne:
        y = [k.get_data()[..., 50:100].sum(axis=(1, 2)) for k in x]
    else:
        y = [k[..., 50:100].sum(axis=(1, 2)) for k in x]
    z = [np.random.randint(0, 10, (len(k), 2)) for k in y]
    x = [k + 1000 for k in x]
    # time -= 5
    # [print(k.shape, i.shape) for k, i in zip(x, y)]

    dt = DatasetEphy(x, y, roi=roi, z=z)
    dt.groupby("roi")
    print([k for k in dt.suj_roi])
    print(dt.shape)
    dt.copnorm(condition="cc", inference="rfx")
    print(dt)
