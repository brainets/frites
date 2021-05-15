"""Auto-regressive models simulations."""
import numpy as np
from scipy import stats
from scipy.signal import periodogram
import xarray as xr

from frites.io import set_log_level, logger
from frites.conn import conn_covgc
from frites.core import mi_model_nd_gd


class StimSpecAR(object):
    """Stimulus-specific autoregressive (AR) Model.

    This class can be used to simulate several networks where the information
    sent between node is stimulus specific inside a temporal region.
    """

    def __init__(self, verbose=None):
        """Init."""
        set_log_level(verbose)

    def fit(self, ar_type='hga', sf=200, n_times=300, n_epochs=100, dt=50,
            n_stim=3, n_std=3, stim_onset=100, random_state=None):
        """Get the data generated by the selected model.

        Parameters
        ----------
        ar_type : {'hga', 'osc_20', 'osc_40', 'ding_2', 'ding_3', 'ding_5'}
            Autoregressive model type. Choose either :

                * 'hga' : for evoked high-gamma activity
                * 'osc_20' / 'osc_40' : for oscillations respectively around
                  20Hz and 40Hz
                * 'osc_40_3' : oscillations at 40hz for 3 nodes. This model
                  simulates X->Y, X->Z and instantaneous Y.Z
                * 'ding_2' / 'ding_3_direct' / 'ding_3_indirect' / 'ding_5' :
                  respectively the models with 2, 3 or 5 nodes described by
                  Ding et al. :cite:`ding2006granger`

        sf : float | 200
            The sampling frequency
        n_times : int | 300
            Number of time points
        n_epochs : int | 100
            Number of epochs
        dt : int | 50
            Width of the time-varying Gaussian stimulus
        n_stim : int | 3
            Number of stimulus to use
        n_std : float, int | 3
            Number of standard deviations the stimulus exceed the random noise.
            Should be an integer striclty over 1. Note that this concerns the
            first stimulus. For example, if n_std=3, the first stimulus is
            going to have a deviation 3 times larger than the noise, the second
            stimulus 6 times the noise, the third stimulus 9 times.
        stim_onset : int | 100
            Index where the time-varying Gaussian stimulus should start
        random_state : int | None
            Fix the random state of the machine for reproducibility

        Returns
        -------
        data : xarray.DataArray
            DataArray of shape (n_epochs * n_stim, n_roi, n_times)
        """
        assert isinstance(n_std, (int, float))
        times = np.arange(n_times) / sf - 0.5
        cval = np.arange(n_stim) + 1
        gval = np.arange(n_stim) + 1
        n_epochs_tot = int(n_epochs * n_stim)
        stim = np.repeat(np.arange(n_stim) + 1, n_epochs)
        if not isinstance(random_state, int):
            random_state = np.random.randint(10000)
        kw_noise = dict(size=(n_epochs_tot, n_times), loc=0,
                        random_state=random_state)

        if ar_type == 'hga':
            self._lab = 'Evoked HGA'
        elif ar_type == 'osc_20':
            self._lab = '20Hz oscillations'
        elif ar_type in ['osc_40', 'osc_40_3']:
            self._lab = '40Hz oscillations'
        elif 'ding' in ar_type:
            self._lab = f"Ding's {ar_type[-1]} nodes"

        logger.info(f"{self._lab} AR model (n_times={n_times}, "
                    f"n_epochs={n_epochs}, n_stim={n_stim}, "
                    f"random_state={random_state})")

        # ---------------------------------------------------------------------
        #                             GAUSSIAN STIM
        # ---------------------------------------------------------------------
        # generate time-varying Gaussian input to X
        gauss = stats.norm.pdf(np.linspace(-5, 5, dt + 1, endpoint=True), 0, 2)
        # normalise Gaussian profile between [0, 1]
        gauss -= gauss.min()
        gauss /= gauss.max()
        # full time gaussian stim
        gauss_stim = np.zeros((n_times,), dtype=float)
        gauss_stim[stim_onset - 1:stim_onset + len(gauss) - 1] = gauss

        # ---------------------------------------------------------------------
        #                            COUPLING STRENGTH
        # ---------------------------------------------------------------------
        c = np.repeat(cval, n_epochs)
        c = c.reshape(-1, 1) * gauss_stim.reshape(1, -1)
        causal = np.array(cval).reshape(-1, 1) * gauss_stim.reshape(1, -1)

        # ---------------------------------------------------------------------
        #                        AUTOREGRESSIVE MODEL
        # ---------------------------------------------------------------------

        if ar_type in ['hga', 'osc_20', 'osc_40']:
            # _____________________________ NOISE _____________________________
            # white noise with zero mean and unit variance
            n1, n2 = self._generate_noise(var=[.05, .05], **kw_noise)

            # _____________________________ GAIN ______________________________
            if ar_type == 'hga':
                # generate the array of gain
                g = np.repeat(gval, n_epochs)
                g = g.reshape(-1, 1) * gauss_stim.reshape(1, -1)
                # modulates gain according to n_std
                g = self._n_std_gain(g, n1, n_std)
                # for hga, there's no need to have an additional modulation
                c = np.ones_like(c)
            else:
                g = np.zeros((n_epochs_tot, n_times), dtype=float)

            # ________________________ N_STD COUPLING _________________________
            c2 = self._n_std_gain(c, n2, n_std)

            # _______________________ POLY COEFFICIENTS________________________
            if ar_type == 'osc_40':    # bivariate data oscillating at 40Hz
                a1 = [.55, -.8]
                a2 = [.35, -.5]
                a12 = [.5, 0.]
            elif ar_type == 'osc_20':  # bivariate data oscillating at ~20Hz
                a1 = [0, .05, .05, 0, -.3, -.3]
                a2 = [0, 0, 0, 0, -.3, -0.3]
                a12 = [0, 0, .5, .5, 0, 0]
            elif ar_type == 'hga':     # Evoked High-Gamma Activity (order 5)
                a1 = [.3]
                a2 = [.3]
                a12 = [0, 0, 0, .5, .5]

            # ______________________________ AR _______________________________
            # generate AR model with feature-specific causal connectivity (fCC)
            order = np.max([len(a1), len(a2), len(a12)])
            x, y = n1, n2
            for t in range(order, n_times):
                # past indexing
                _sl_a1 = np.arange(t - 1, t - len(a1) - 1, -1)
                _sl_a2 = np.arange(t - 1, t - len(a2) - 1, -1)
                _sl_a12 = np.arange(t - 1, t - len(a12) - 1, -1)
                # AR core
                # - x1 = noise + gain + a1 * past_x1
                # - x2 = noise + a2 * past_x2 + coupling * past_x1
                x[:, t] = n1[:, t] + g[:, t] + (x[:, _sl_a1] @ a1)
                y[:, t] = n2[:, t] + (y[:, _sl_a2] @ a2) + c2[:, t] * (
                    x[:, _sl_a12] @ a12)
            # concatenate everything
            dat, roi = np.stack((x, y), axis=1), ['x', 'y']
        elif ar_type == 'osc_40_3':
            n1, n2, n3 = self._generate_noise(var=[.05] * 3, **kw_noise)
            c2 = self._n_std_gain(c, n2, n_std)
            c3 = self._n_std_gain(c, n3, n_std)

            x, y, z = n1, n2, n3
            for t in range(2, n_times):
                x[:, t] = n1[:, t] + .55 * x[:, t - 1] - .8 * x[:, t - 2]
                y[:, t] = n2[:, t] + .35 * y[:, t - 1] - .5 * y[:, t - 2] + (
                    c2[:, t] * (.5 * x[:, t - 1]))
                z[:, t] = n3[:, t] + .35 * z[:, t - 1] - .5 * z[:, t - 2] + (
                    c3[:, t] * (.5 * x[:, t - 1]))
            dat, roi = np.stack((x, y, z), axis=1), ['x', 'y', 'z']
        elif ar_type == 'ding_2':
            n1, n2 = self._generate_noise(var=[1., .7], **kw_noise)
            c2 = self._n_std_gain(c, n2, n_std)

            x, y = n1, n2
            for t in range(2, n_times):
                x[:, t] = .9 * x[:, t - 1] - .5 * x[:, t - 2] + n1[:, t]
                y[:, t] = .8 * y[:, t - 1] - .5 * y[:, t - 2] + c2[:, t] * (
                    .16 * x[:, t - 1] - .2 * x[:, t - 2]) + n2[:, t]
            dat, roi = np.stack((x, y), axis=1), ['x', 'y']
        elif ar_type in ['ding_3_direct', 'ding_3_indirect']:
            n1, n2, n3 = self._generate_noise(var=[.3, 1., .2], **kw_noise)
            c1 = self._n_std_gain(c, n1, n_std)
            c2 = self._n_std_gain(c, n2, n_std)
            c3 = self._n_std_gain(c, n3, n_std)

            x, y, z = n1, n2, n3
            for t in range(2, n_times):
                if ar_type == 'ding_3_indirect':
                    x[:, t] = .8 * x[:, t - 1] - .5 * x[:, t - 2] + c1[
                        :, t] * (.4 * z[:, t - 1]) + n1[:, t]
                elif ar_type == 'ding_3_direct':
                    x[:, t] = .8 * x[:, t - 1] - .5 * x[:, t - 2] + c1[
                        :, t] * (.4 * z[:, t - 1] + .2 * y[:, t - 2]) + n1[
                        :, t]
                y[:, t] = .9 * y[:, t - 1] - .8 * y[:, t - 2] + n2[:, t]
                z[:, t] = .5 * z[:, t - 1] - .2 * z[:, t - 2] + c3[:, t] * (
                    .5 * y[:, t - 1]) + n3[:, t]
            dat, roi = np.stack((x, y, z), axis=1), ['x', 'y', 'z']
            ar_type = 'ding_3'
        elif ar_type == 'ding_5':
            n1, n2, n3, n4, n5 = self._generate_noise(var=[.6, .5, .3, .3, .6],
                                                      **kw_noise)
            c2 = self._n_std_gain(c, n2, n_std)
            c3 = self._n_std_gain(c, n3, n_std)
            c4 = self._n_std_gain(c, n4, n_std)
            c5 = self._n_std_gain(c, n5, n_std)
            sq2 = np.sqrt(2.)

            x1, x2, x3, x4, x5 = n1, n2, n3, n4, n5
            for t in range(3, n_times):
                x1[:, t] = .95 * sq2 * x1[:, t - 1] - .9025 * x1[
                    :, t - 2] + n1[:, t]
                x2[:, t] = c2[:, t] * (.5 * x1[:, t - 2]) + n2[:, t]
                x3[:, t] = c3[:, t] * (-.4 * x1[:, t - 3]) + n3[:, t]
                x4[:, t] = c4[:, t] * (-.5 * x1[:, t - 2] + .25 * sq2 * x5[
                    :, t - 1]) + .25 * sq2 * x4[:, t - 1] + n4[:, t]
                x5[:, t] = c5[:, t] * (-.25 * sq2 * x4[:, t - 1]) + (
                    .25 * sq2 * x5[:, t - 1] + n5[:, t])
            dat = np.stack((x1, x2, x3, x4, x5), axis=1)
            roi = ['x1', 'x2', 'x3', 'x4', 'x5']

        # ---------------------------------------------------------------------
        #                         XARRAY CONVERSION
        # ---------------------------------------------------------------------
        ar = xr.DataArray(dat, dims=('trials', 'roi', 'times'),
                          coords=(stim, np.array(roi), times))
        # add attributes
        attr = dict(n_stim=n_stim, n_std=n_std, ar_type=ar_type, stimulus=cval)
        for k, v in attr.items(): ar.attrs[k] = v  # noqa
        # keep in object
        self._ar_type = ar_type
        self._causal = causal
        self._ar = ar
        self._sf = sf
        self._n_stim = n_stim
        self._n_std = n_std

        return ar

    @staticmethod
    def _generate_noise(size=(1,), loc=0, var=1, random_state=0):
        """Generate random gaussian noise."""
        if not isinstance(var, (list, tuple, np.ndarray)):
            var = [var]
        n = []
        for n_k, k in enumerate(var):
            rnd = np.random.RandomState(random_state + n_k)
            n += [rnd.normal(scale=np.sqrt(k), loc=loc, size=size)]
        return tuple(n)

    @staticmethod
    def _n_std_gain(c, noise, n_std):
        """Adapt the coupling strength to the deviation of the noise."""
        return c * (n_std * noise.std())

    def _compute_psd(self, x):
        """Compute the stimulus specific PSD of a single roi.

        `x` should be a dataarray of shape (n_epochs, n_roi, n_times)
        """
        # compute psd
        n_times = len(self._ar['times'])
        freqs, psd = periodogram(
            x.data, fs=self._sf, window=None, nfft=n_times, detrend='constant',
            return_onesided=True, scaling='density', axis=2)
        # dataarray conversion
        psd = xr.DataArray(psd, dims=('trials', 'roi', 'freqs'),
                           coords=(self._ar['trials'], self._ar['roi'], freqs))
        return psd

    ###########################################################################
    ###########################################################################
    #                              COVGC
    ###########################################################################
    ###########################################################################

    def compute_covgc(self, ar, dt=50, lag=5, step=1, method='gc',
                      conditional=False):
        """Compute the Covariance-based Granger Causality.

        In addition of computing the Granger Causality, the mutual-information
        between the Granger causalitity and the stimulus is also computed.

        Parameters
        ----------
        dt : int
            Duration of the time window for covariance correlation in samples
        lag : int
            Number of samples for the lag within each trial
        step : int | 1
            Number of samples stepping between each sliding window onset
        method : {'gauss', 'gc'}
            Method for the estimation of the covgc. Use either 'gauss' which
            assumes that the time-points are normally distributed or 'gc' in
            order to use the gaussian-copula.

        Returns
        -------
        gc : array_like
            Granger Causality arranged as (n_epochs, n_pairs, n_windows, 3)
            where the last dimension means :

                * 0 : pairs[:, 0] -> pairs[:, 1] (x->y)
                * 1 : pairs[:, 1] -> pairs[:, 0] (y->x)
                * 2 : instantaneous  (x.y)
        """
        # compute the granger causality
        t0 = np.arange(lag, ar.shape[-1] - dt, step)
        gc = conn_covgc(ar, dt, lag, t0, times='times', method=method,
                        roi='roi', step=1, conditional=conditional)
        gc['trials'] = ar['trials']
        self._gc = gc
        # compute the MI between stimulus / raw
        mi = mi_model_nd_gd(gc.data, gc['trials'].data, traxis=0)
        self._mi = xr.DataArray(mi, dims=('roi', 'times', 'direction'),
                                coords=(gc['roi'], gc['times'],
                                        gc['direction']))
        self._mi.attrs['description'] = (
            "Mutual information between the stimulus and the output of the "
            "covgc")

        return gc

    ###########################################################################
    ###########################################################################
    #                              PLOTTING
    ###########################################################################
    ###########################################################################

    def plot(self, psd=False, cmap='plasma', colorbar=False, **kwargs):
        """Plot the generated data.

        Parameters
        ----------
        psd : bool | False
            If False (default), the raw data are plotted. If True, the power
            spectrum density (PSD) is plotted instead
        cmap : 'string' | 'plasma'
            Colormap to use
        colorbar : bool | False
            Display or not the colorbar
        kwargs : dict | {}
            Additional inputs are sent to the `plt.imshow` function
        """
        import matplotlib.pyplot as plt

        times = self._ar.times.data
        trials = np.arange(len(self._ar.trials))
        n_roi = len(self._ar.roi.data)
        # switch between raw / psd plot
        if not psd:
            to_plt, xaxis, ext = self._ar, times, ''
        else:
            to_plt = self._compute_psd(self._ar)
            xaxis, ext = to_plt['freqs'], 'PSD of '

        kw_imshow = dict(extent=[xaxis[0], xaxis[-1], trials[0], trials[-1]],
                         aspect='auto', origin='lower', cmap=cmap, **kwargs)

        # plot stimulus
        plt.subplot(n_roi + 1, 1, 1)
        plt.plot(times, self._causal.T)
        plt.ylabel('Causal coupling'), plt.xlabel('Time (seconds)')
        plt.title(r"Causal coupling from X $\rightarrow$ Y for different "
                  "stims", fontweight='bold')
        plt.grid(True)
        plt.axvline(0, lw=2, color='k')
        plt.xlim(times[0], times[-1])
        # plot raw / psd ( new style rules :) )
        for n_r in range(n_roi):
            plt.subplot(n_roi + 1, 1, 2 + n_r)
            plt.imshow(to_plt.isel(roi=n_r), **kw_imshow)
            plt.ylabel('Trials')
            plt.title(
                f"Single trial {ext}{str(to_plt['roi'].data[n_r]).upper()}",
                fontweight='bold')
            if not psd:
                plt.axvline(0, lw=2, color='w')
            if n_r != n_roi - 1:
                plt.xlabel('')
                plt.tick_params(labelbottom=False, bottom=False)
            else:
                lab = 'Frequencies (Hz)' if psd else 'Time (seconds)'
                plt.xlabel(lab)
            if colorbar:
                plt.colorbar()

        return plt.gca()

    def plot_model(self):
        """Plot the model of the network.

        Note that this method requires the networkx Python package.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph(directed=True)
        if self._ar_type in ['hga', 'osc_40', 'osc_20']:
            G.add_edges_from([('X', 'Y')], weight=1)
            lab = self._lab
            edge_labels = {
                ('X', 'Y'): lab + f"\n(n_stim={self._n_stim}, "
                                  f"n_std={self._n_std})"}
        elif self._ar_type == 'osc_40_3':
            G.add_edges_from([('X', 'Y')], weight=1)
            G.add_edges_from([('X', 'Z')], weight=1)
            lab = self._lab
            edge_labels = {(u, v): rf"{u}$\rightarrow${v}={d['weight']}"
                           for u, v, d in G.edges(data=True)}
        elif self._ar_type in ['ding_2', 'ding_3', 'ding_5']:
            if self._ar_type == 'ding_2':
                G.add_edges_from([('X', 'Y')], weight=1)
            elif self._ar_type == 'ding_3':
                G.add_edges_from([('Y', 'X')], weight=5)
                G.add_edges_from([('Y', 'Z')], weight=6)
                G.add_edges_from([('Z', 'X')], weight=4)
            elif self._ar_type == 'ding_5':
                G.add_edges_from([('X1', 'X2'), ('X1', 'X3'), ('X1', 'X4')],
                                 weight=5)
                G.add_edges_from([('X4', 'X5')], weight=2)
                G.add_edges_from([('X5', 'X4')], weight=1)
            # build edges labels
            edge_labels = {(u, v): rf"{u}$\rightarrow${v}={d['weight']}"
                           for u, v, d in G.edges(data=True)}
            # fix ding_5 for bidirectional connectivity between 4 <-> 5
            if self._ar_type == 'ding_3':
                edge_labels[('Y', 'X')] = "indirect\n(mediated by Z)"
            elif self._ar_type == 'ding_5':
                edge_labels[('X5', 'X4')] = (r"X4$\rightarrow$X5=2" + "\n" +
                                             r'X5$\rightarrow$X4=1')

        # color edges according to causal strength
        colors = np.array([w['weight'] for _, _, w in G.edges(data=True)])
        # get edge labels in the form {('X', 'Y'): 1 etc.}
        pos = nx.planar_layout(G)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.draw_networkx(G, pos, arrowstyle='-|>', arrowsize=25, node_size=800,
                         arrows=True, font_color='w', node_color='k',
                         edge_cmap=plt.cm.Greys, edge_color=colors, width=3,
                         edge_vmin=colors.min() - 1)
        plt.axis('off')
        plt.tight_layout()

        return plt.gca()

    def plot_covgc(self, gc=None, plot_mi=False):
        """Plot the Granger Causality.

        Note that before plotting the Granger causality, the method
        :class:`StimSpecAR.compute_covgc` have to be launched before.

        Parameters
        ----------
        gc : DataArray | None
            Granger Causality output of the function
            :func:`frites.conn.conn_covgc`
        plot_mi : bool | False
            If False (default) the Granger causality is plotted. If True, it is
            the information shared between the Granger causality and the
            stimulus that is plotted.
        """
        import matplotlib.pyplot as plt

        if isinstance(gc, xr.DataArray):
            gc['trials'] = self._ar['trials']
            self._gc = gc
        # select either the covgc either the mi(covgc; stim)
        if not plot_mi:
            gcm = self._gc.groupby('trials').mean('trials')
            gcm = gcm.rename({'trials': 'Stimulus'})
        else:
            gcm = self._mi
        # conditional covgc
        ext = '|others' if self._gc.attrs['conditional'] else ''

        y_min, y_max = gcm.data.min(), gcm.data.max()
        direction, roi = gcm['direction'].data, gcm['roi'].data
        q = 1
        for n_d, d in enumerate(direction):
            for n_r, r in enumerate(roi):
                plt.subplot(len(direction), len(roi), q)
                gcm.sel(roi=r, direction=d).plot.line(
                    x='times', hue='Stimulus', add_legend=False)
                plt.ylim(y_min, y_max)
                plt.xlim(gcm['times'][0], gcm['times'][-1])
                plt.axvline(0., lw=2., color='k')
                r_sp = r.split('-')
                if d == 'x->y':
                    tit = fr'{r_sp[0]}$\rightarrow${r_sp[1]}{ext}'
                elif d == 'y->x':
                    tit = fr'{r_sp[1]}$\rightarrow${r_sp[0]}{ext}'
                elif d == 'x.y':
                    tit = fr'{r_sp[0]} . {r_sp[1]}{ext}'
                plt.title(tit, fontweight='bold', fontsize=15)
                if n_r >= 1:
                    plt.ylabel('')
                    plt.tick_params(labelleft=False, left=False)
                if n_d < len(direction) - 1:
                    plt.xlabel('')
                    plt.tick_params(labelbottom=False, bottom=False)
                q += 1

        return plt.gca()

    ###########################################################################
    ###########################################################################
    #                              PROPERTIES
    ###########################################################################
    ###########################################################################

    @property
    def ar(self):
        """Output data generated by the selected model."""
        return self._ar

    @property
    def gc(self):
        """Granger causality."""
        return self._gc

    @property
    def mi(self):
        """Mutual-information between the granger causality and stimulus."""
        return self._mi


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ss = StimSpecAR()
    ar = ss.fit(ar_type='hga', random_state=1, n_std=1, n_stim=2,
                n_epochs=20)
    mi = mi_model_nd_gd(ar.data, ar['trials'].data, traxis=0)
    plt.plot(mi.T)
    plt.show()
    exit()
    # ss.plot(cmap='viridis', psd=False, colorbar=True)
    # plt.show()
    # ss.plot_model()
    gc = ss.compute_covgc(ar, step=1, conditional=False)
    # plt.figure(figsize=(14, 12))
    # ss.plot_covgc(plot_mi=True)
    # plt.figure(figsize=(14, 12))
    # ss.plot_covgc(plot_mi=True)
    plt.tight_layout()
    plt.show()
