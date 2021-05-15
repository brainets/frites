"""Multi-subjects container of neurophysiological."""
import numpy as np
import xarray as xr
import pandas as pd

import frites
from frites.config import CONFIG
from frites.io import set_log_level, logger, Attributes
from frites.dataset import SubjectEphy
from frites.dataset.ds_utils import multi_to_uni_conditions
from frites.core import copnorm_cat_nd, copnorm_nd
from frites.conn.conn_utils import conn_get_pairs
from frites.utils import savgol_filter, nonsorted_unique


class DatasetEphy(object):
    """Multi-subjects electrophysiological data container.

    This class is a container used to represent the neurophysiological data
    coming from multiple subjects. In addition to passing the data, this
    container is also going to need the anatomical information such as the
    task-related variables (continuous or discret). Then, the created object
    can be used to compute the mutual information (MI) between the data and the
    task variable and group-level.

    Parameters
    ----------
    x : list
        List of length (n_subjects,) where each element of the list should be
        the neural data of single subject. Supported types for each element of
        the list are the same as in :class:`SubjectEphy`
    y, z : list, sting | None
        List of length (n_subjects,) of continuous or discrete task-related
        variables per subject. Supported types for each element of the list are
        the same as in :class:`SubjectEphy`
    roi : list | None
        List of length (n_subjects,) where each element represents the
        anatomical information of each channel. Supported types for each
        element of the list are the same as in :class:`SubjectEphy`
    agg_ch : bool | True
        If multiple channels belong to the same ROI, specify whether if the
        data should be aggregated across channels (True) or if the information
        per channel have to be take into account (False - conditional mutual
        information).
    times : array_like | None
        The time vector to use. Supported types are listed in
        :class:`SubjectEphy`
    multivariate : bool | False
        If 4d input is provided, this parameter specifies whether this axis
        should be considered as multi-variate (True) or uni-varariate (False)
    nb_min_suj : int | None
        The minimum number of subjects per roi. Roi with n_suj < nb_min_suj
        are going to be skipped. Use None to force selecting all of the
        subjects
    attrs : dict | {}
        Dictionary of additional attributes about the data
    """

    def __init__(self, x, y=None, z=None, roi=None, agg_ch=True, times=None,
                 multivariate=False, nb_min_suj=False, attrs=None,
                 verbose=None):
        """Init."""
        set_log_level(verbose)
        self.attrs = Attributes(attrs=attrs)
        assert isinstance(x, (list, tuple))
        self._agg_ch = agg_ch
        self._multivariate = multivariate

        logger.info('Definition of an electrophysiological dataset')
        logger.info(f'    Dataset composed of {len(x)} subjects / sessions')

        # ========================== Multi-conditions =========================

        # remapping group y and z
        if isinstance(y, (list, tuple)):
            y = multi_to_uni_conditions(y, var_name='y', verbose=verbose)
        if isinstance(z, (list, tuple)):
            z = multi_to_uni_conditions(z, var_name='z', verbose=verbose)

        # ===================== Multi-subjects conversion =====================

        # force converting the data (latest task-related variables)
        n_subjects = len(x)
        y = [y] * n_subjects if not isinstance(y, list) else y
        z = [z] * n_subjects if not isinstance(z, list) else z
        roi = [roi] * n_subjects if not isinstance(roi, list) else roi
        for k in range(n_subjects):
            x[k] = SubjectEphy(
                x[k], y=y[k], z=z[k], roi=roi[k], agg_ch=True, times=times,
                multivariate=multivariate, verbose=verbose)
        self._x = x

        # minimum number of subject / roi
        nb_min_suj = -np.inf if not isinstance(nb_min_suj, int) else nb_min_suj
        self._nb_min_suj = nb_min_suj
        logger.info(f"    At least {self._nb_min_suj} subjects / roi required")

        # merge attributes
        self.attrs.merge([k.attrs for k in self._x])
        self._y_dtype = self.attrs['y_dtype']
        self._z_dtype = self.attrs['z_dtype']
        self._mi_type = self.attrs['mi_type']
        mi_repr = self.attrs['mi_repr']
        logger.info(f"    Supported MI definition {mi_repr} ({self._mi_type})")

        # ===================== Additional dimensions  ========================

        # Subject dimension
        for n_k, k in enumerate(range(len(self._x))):
            self._x[k].name = f'subject_{n_k}'
            self._x[k] = self._x[k].assign_coords(
                subject=('trials', [n_k] * self._x[k].shape[0]))
        # channel aggregation
        if not agg_ch:
            # split into sections of unique intergers
            n_trials_s = [k.shape[1] for k in self._x]
            agg_ch_num = np.arange(np.sum(n_trials_s))
            agg_split = np.split(agg_ch_num, np.cumsum(n_trials_s)[0:-1])
            # add additional dimension
            for k in range(len(self._x)):
                self._x[k] = self._x[k].assign_coords(
                    agg_ch=('roi', agg_split[k]))
        # final mi dimension
        dims = list(self._x[0].dims)
        self._mi_dims = [k for k in dims if k not in ['trials', 'mv']]

        # ============================= Attributes ============================

        # update internals parameters
        self._update_internals()
        # # update internal attributes
        self.attrs.update({
            'nb_min_suj': nb_min_suj,
            'n_subjects': len(self._x),
            'agg_ch': agg_ch,
            'multivariate': multivariate,
            'dtype': "DatasetEphy",
            '__version__': frites.__version__
        })

    ###########################################################################
    ###########################################################################
    #                               INTERNALS
    ###########################################################################
    ###########################################################################

    def __repr__(self):
        """String representation."""
        dt = []
        for k in self._x:
            dt += [repr(k)]
        return '\n'.join(dt)

    def _repr_html_(self):
        from xarray.core.formatting_html import collapsible_section
        dt = []
        for k in self._x:
            dt += [collapsible_section(k.name, details=k._repr_html_(),
                                       n_items=1, collapsed=True)]
        return "".join(f"<li class='xr-section-item'>{s}</li>" for s in dt)

    def _update_internals(self):
        """Update internal variables."""
        # build a unique list of unsorted roi names
        merged_roi = np.r_[tuple([k['roi'].data for k in self._x])]
        roi_names = nonsorted_unique(merged_roi)

        # dataframe made of unique roi per subjects and subject id
        suj_r, roi_r = [], []
        for k in range(len(self._x)):
            _roi = np.unique(self._x[k]['roi'].data).tolist()
            roi_r += _roi
            suj_r += [k] * len(_roi)
        df_rs = pd.DataFrame({'roi': roi_r, '#subjects': suj_r})

        # get number and id of subjects per roi
        gp_roi = df_rs.groupby('roi')
        groups = gp_roi.indices
        u_suj = [list(df_rs.loc[groups[k], '#subjects']) for k in roi_names]
        df_rs = gp_roi.count().reindex(roi_names)
        df_rs['subjects'] = u_suj
        df_rs['keep'] = df_rs['#subjects'] >= self._nb_min_suj

        self._df_rs = df_rs
        self._times = self._x[0]['times'].data
        self._n_times = len(self._times)
        self._n_subjects = len(self._x)
        self._roi_names = list(df_rs.index[df_rs['keep']])
        self._n_roi = len(self._roi_names)

    ###########################################################################
    ###########################################################################
    #                                METHODS
    ###########################################################################
    ###########################################################################

    def get_roi_data(self, roi, groupby='subjects', mi_type='cc', copnorm=True,
                     gcrn_per_suj=True):
        """Get the data of a single brain region.

        Parameters
        ----------
        roi : string
            ROI name to get
        groupby : {'subjects'}
            Specify if the data across subjects have to be concatenated
        mi_type : {'cc', 'cd', 'ccd'}
            The type of mutual-information that is then going to be used. This
            is going to have an influence on how the data are organized and
            how the copnorm is going to be applied
        copnorm : bool | True
            Apply the gaussian copula rank normalization
        gcrn_per_suj : bool | True
            Specify whether the gaussian copula rank normalization have to be
            applied per subject (True - RFX) or across subjects (False - FFX)

        Returns
        -------
        da : xr.DataArray
            The data of the single brain region
        """
        # list of subjects present in the desired roi
        suj_list = self._df_rs.loc[roi, 'subjects']

        # group data across subjects
        if groupby == 'subjects':
            x_r_ms = []
            for s in suj_list:
                # roi (possibly multi-sites) selection
                x_roi = self._x[s].sel(roi=self._x[s]['roi'].data == roi)
                # stack roi and trials
                x_roi = x_roi.stack(rtr=('roi', 'trials'))
                x_r_ms.append(x_roi)
            x_r_ms = xr.concat(x_r_ms, 'rtr')
            # 4d or multivariate
            if self._multivariate:
                x_r_ms = x_r_ms.transpose('times', 'mv', 'rtr')
            else:
                x_r_ms = x_r_ms.expand_dims('mv', axis=-2)
            x_coords = list(x_r_ms.coords)

            # channels aggregation
            if not self._agg_ch and ('y' in x_coords):
                # shortcuts
                ch_id = x_r_ms['agg_ch'].data
                y = x_r_ms['y'].data
                # transformation depends on mi_type
                if mi_type == 'cd':
                    # I(C; D) where the D=[y, ch_id]
                    ysub = np.c_[y, ch_id]
                    x_r_ms['y'].data = multi_to_uni_conditions(
                        [ysub], False)[0]
                elif (mi_type == 'ccd') and ('z' not in x_coords):
                    # I(C; C; D) where D=ch_id. In that case z=D
                    x_r_ms = x_r_ms.assign_coords(z=('rtr', ch_id))
                elif (mi_type == 'ccd') and ('z' in x_coords):
                    # I(C; C; D) where D=[z, ch_id]
                    zsub = np.c_[x_r_ms['z'].data, ch_id]
                    x_r_ms['z'].data = multi_to_uni_conditions(
                        [zsub], False)[0]
                else:
                    raise ValueError("Can't avoid aggregating channels")

            # gaussian copula rank normalization
            if copnorm:
                if gcrn_per_suj:  # gcrn per subject
                    logger.debug("copnorm applied per subjects")
                    suj = x_r_ms['subject'].data
                    x_r_ms.data = copnorm_cat_nd(x_r_ms.data, suj, axis=-1)
                    if (mi_type in ['cc', 'ccd']) and ('y' in x_coords):
                        x_r_ms['y'].data = copnorm_cat_nd(
                            x_r_ms['y'].data, suj, axis=0)
                else:             # gcrn across subjects
                    logger.debug("copnorm applied across subjects")
                    x_r_ms.data = copnorm_nd(x_r_ms.data, axis=-1)
                    if (mi_type in ['cc', 'ccd']) and ('y' in x_coords):
                        x_r_ms['y'].data = copnorm_nd(x_r_ms['y'].data, axis=0)

            return x_r_ms

    def sel(self, **kwargs):
        """Coordinate-based data slicing.

        Slice the entire dataset based on the coordinates values.

        Parameters
        ----------
        kwargs : {} | None
            Additional inputs are sent to to the method `xr.DataArray.sel` of
            the data of each subject.

        Returns
        -------
        inst : instance of DatasetEphy
            The sliced object.

        Examples
        --------
        > # define the dataset
        > ds = DatasetEphy(...)
        > # temporal slicing of the data between (-100ms, 1500ms)
        > ds.sel(times=slice(-0.1, 1.5))
        """
        self._x = [k.sel(**kwargs) for k in self._x]
        self._update_internals()
        return self

    def isel(self, **kwargs):
        """Index-based data slicing.

        Slice the entire dataset based on indexes.

        Parameters
        ----------
        kwargs : {} | None
            Additional inputs are sent to to the method `xr.DataArray.isel` of
            the data of each subject.

        Returns
        -------
        inst : instance of DatasetEphy
            The sliced object.

        Examples
        --------
        > # define the dataset
        > ds = DatasetEphy(...)
        > # temporal slicing of the data between time points (100, 2500)
        > ds.sel(times=slice(100, 2500))
        """
        self._x = [k.isel(**kwargs) for k in self._x]
        self._update_internals()
        return self

    def savgol_filter(self, h_freq, edges=None, verbose=None):
        """Filter the data using Savitzky-Golay polynomial method.

        This method is an adaptation of the mne-python one. Note that this
        smoothing operation is performed inplace to avoid data copy.

        Parameters
        ----------
        h_freq : float
            Approximate high cut-off frequency in Hz. Note that this is not an
            exact cutoff, since Savitzky-Golay filtering is done using
            polynomial fits instead of FIR/IIR filtering. This parameter is
            thus used to determine the length of the window over which a
            5th-order polynomial smoothing is used.
        edges : int, float | None
            Edge compensation. Use either an integer to drop a specific number
            of time points (e.g edges=100 remove 100 time points at the
            begining and at the end) or a float to drop a period (e.g
            edges=0.2 drop 200ms at the begining and at the end)

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

        # perform smoothing
        for n_s in range(len(self._x)):
            self._x[n_s] = savgol_filter(self._x[n_s], h_freq, axis='times',
                                         sfreq=self.attrs['sfreq'],
                                         verbose=verbose)
        # edge effect compensation
        if isinstance(edges, CONFIG['FLOAT_DTYPE']):
            t = self._times
            self.sel(times=slice(t[0] + edges, t[-1] - edges))
        elif isinstance(edges, CONFIG['INT_DTYPE']):
            self.isel(times=slice(edges, -edges))

        return self

    def get_connectivity_pairs(self, as_blocks=False, directed=False,
                               verbose=None):
        """Get the connectivity pairs for this dataset.

        This method can be used to get the possible connectivity pairs i.e
        (sources, targets) for directed connectivity (or not). In addition,
        some pairs are going to be ignored because of a number of subjects to
        low.

        Parameters
        ----------
        directed : bool | False
            Get either directed (True) or non-directed (False) pairs

        Returns
        -------
        df_conn : pd.DataFrame
            The table describing the connectivity informations per pair of
            brain regions
        df_conn_suj : pd.DataFrame
            The table describing the connectivity informations per subject
        """
        rois = [k['roi'].data.astype(str) for k in self.x]
        # get the dataframe for connectivity
        self.df_conn, self.df_conn_suj = conn_get_pairs(
            rois, directed=directed, nb_min_suj=self._nb_min_suj,
            verbose=verbose)
        # filter both dataframes
        df_conn = self.df_conn.loc[self.df_conn['keep']]
        df_conn = df_conn.drop(columns='keep')
        df_conn = df_conn.reset_index(drop=True)
        df_conn_suj = self.df_conn_suj.loc[self.df_conn_suj['keep_suj']]
        df_conn_suj = df_conn_suj.drop(columns='keep_suj')
        df_conn_suj = df_conn_suj.reset_index(drop=True)

        # group by sources
        if as_blocks:
            df_conn = df_conn.groupby('sources').agg(list).reset_index()

        return df_conn, df_conn_suj

    ###########################################################################
    ###########################################################################
    #                               PROPERTIES
    ###########################################################################
    ###########################################################################

    @property
    def x(self):
        """Multi-subjects electrophysiological data (DataArray)."""
        return self._x

    @property
    def df_rs(self):
        """Pandas DataFrame of cross-subjects anatomical repartition."""
        return self._df_rs

    @property
    def times(self):
        """Time vector."""
        return self._times

    @property
    def roi_names(self):
        """List of brain regions to keep across subjects."""
        return self._roi_names
