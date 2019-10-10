"""Dataset representation of electrophysiological data."""
import logging

import numpy as np

import frites
from frites.config import CONFIG

logger = logging.getLogger("frites")


class DatasetEphy(object):
    """Dataset of electrophysiological data coming from several subjects.

    The following steps are performed :

        * Load the data of each subject (support for MNE types) and convert
          into NumPy arrays
        *

    Parameters
    ----------
    """

    def __init__(self, x, y, z=None, roi=None, times=None, nb_min_suj=None):
        """Init."""
        # ---------------------------------------------------------------------
        # check input

        assert all([isinstance(k, (list, tuple)) for k in (x, y)])
        assert len(x) == len(y)

        self.nb_min_suj = nb_min_suj
        self.n_subjects = len(x)
        self.times = times

        logger.info(f"Creation of a dataset composed with {self.n_subjects} "
                    f"subjects. A minimum of {self.nb_min_suj} per roi is"
                    "required")

        # internals
        self.modality = "electrophysiological"
        self._copnormed = False
        self._groupedby = "subject"
        self.__version__ = frites.__version__

        # ---------------------------------------------------------------------
        # load the data of each subject

        logger.info("    Load the data of each subject")
        self._x = [self._load_single_suj_ephy(x[k]) for k in range(len(self))]
        # check the time vector
        _x_times = np.unique([self._x[k].shape[1] for k in range(len(self))])
        assert _x_times.size == 1, ("Inconsistent number of time points across"
                                    " subjects")
        self.n_times = self._x[0].shape[1]
        if not isinstance(self.times, np.ndarray):
            self.times = np.arange(self.n_times)

        # ---------------------------------------------------------------------
        # reorganize by roi
        self.groupby("roi")

    ###########################################################################
    # INTERNALS
    ###########################################################################

    def __repr__(self):
        """String representation."""
        sep = '-' * 79
        rep = (f"{sep}\n"
               f"number of subjects : {self.n_subjects}\n"
               f"minimum number of subject per roi: {self.nb_min_suj}\n"
               f"modality : {self.modality}\n"
               f"data grouped by : {self._groupedby}\n"
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

        # data.reshape(n_roi, n_times, n_trials)
        return np.moveaxis(data, 0, -1)

    def groupby(self, groupby="roi"):
        """Reorganize the data.

        Parameters
        ----------
        groupby : {"roi", "subject"}
            Group data either by subject or by roi
        """
        assert groupby in ["subject", "roi"]
        if groupby == self._groupedby:
            logger.warning("Grouping ignored because already grouped by "
                           f"{self._groupedby}")
            exit()
        logger.info(f"    Group data by {groupby}")

        if groupby == "roi":
            pass
        elif groupby == "subject":
            pass

        self._groupedby = groupby

    def copnorm(self):
        pass

    def save(self):
        pass

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
    def nb_min_suj(self):
        """Get the minimum number of subjects needed per roi."""
        return self._nb_min_suj
    
    @nb_min_suj.setter
    def nb_min_suj(self, value):
        """Set nb_min_suj value."""
        self._nb_min_suj = -np.inf if not isinstance(value, int) else value
    
    
    
    


if __name__ == '__main__':
    x = [0, 1, 2]
    y = [10, 1, 23]

    dt = DatasetEphy(x, y, nb_min_suj=None)
    print(str(dt))
