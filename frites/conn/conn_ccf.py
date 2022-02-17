"""Cross-correlation function."""
import numpy as np
import xarray as xr

from frites.conn import conn_io
from frites.io import logger, check_attrs
from frites.utils import parallel_func
from frites.utils.preproc import _acf


def para_fun(xs, xt):
    """Single-trial ccf on a pair of signals."""
    n_trials, n_times = xs.shape
    corr = np.zeros((n_trials, int(2 * n_times) - 1))
    for n_t in range(n_trials):
        corr[n_t, :] = _acf(xs[n_t, :], xt[n_t, :])
    return corr


def conn_ccf(data, times=None, roi=None, normalized=True, n_jobs=1,
             times_as_sample=True, sfreq=None, verbose=None, **kw_links):
    """Single trial Cross-Correlation Function.

    This function computes the pairwise Cross Correlation (CCF) at the single
    trial level. This can be particulary usefull to find whether there are
    temporal delays between times series.

    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported :

            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)

    times : array_like | None
        Time vector array of shape (n_times,). If the input is an xarray, the
        name of the time dimension can be provided
    roi : array_like | None
        ROI names of a single subject. If the input is an xarray, the
        name of the ROI dimension can be provided
    normalized : bool | True
        Z-score normalization of the data. By default, it set to true.
    times_as_sample : bool | True
        Specify whether the time dimension of the cross-correlation output
        should be described using the time unit of the input data or in
        samples. By default, samples are used to describe lags between
        time-series.
    n_jobs : int | 1
        Number of jobs to use for parallel computing (use -1 to use all
        jobs). The parallel loop is set at the pair level.
    sfreq : float | None
        The sampling frequency.
    kw_links : dict | {}
        Additional arguments for selecting links to compute are passed to the
        function :func:`frites.conn.conn_links`

    Returns
    -------
    ccf : array_like
        The Cross-Correlation array of shape (n_epochs, n_pairs, n_times). When
        the peak of correlation occurs at a negative time it means that the
        target has to be moved **toward** the source. On the contrary, if the
        peak occurs at positive time it means that the target is moved **away**
        of the source.

    See also
    --------
    conn_links
    """
    # ________________________________ INPUTS _________________________________
    # inputs conversion
    kw_links.update({'directed': False, 'net': False})
    data, cfg = conn_io(
        data, times=times, roi=roi, agg_ch=False, win_sample=None,
        name='CCF', verbose=verbose, kw_links=kw_links
    )

    # extract variables
    x, trials, attrs = data.data, data['y'].data, cfg['attrs']
    x_s, x_t = cfg['x_s'], cfg['x_t']
    roi_p, times = cfg['roi_p'], data['times'].data
    n_pairs = len(x_s)

    # data normalization
    if normalized:
        x = (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)

    # __________________________________ CCF __________________________________
    # prepare parallel function
    n_jobs = 1 if n_pairs == 1 else n_jobs
    parallel, p_fun = parallel_func(para_fun, n_jobs=n_jobs, verbose=verbose,
                                    total=n_pairs, mesg='Estimating CCF')

    logger.info(f'Computing CCF between {n_pairs} pairs')

    # compute ccf
    ccf = parallel(
        p_fun(x[:, i_s, :], x[:, i_t, :]) for i_s, i_t in zip(x_s, x_t))
    ccf = np.stack(ccf, axis=1)

    # ________________________________ OUTPUTS ________________________________
    # dataarray conversion
    times_n = np.arange(ccf.shape[-1]).astype(float)
    times_n -= times_n.mean()
    unit = 'samples'
    if not times_as_sample:
        times_n /= cfg['sfreq']
        unit = 'times'
    ccf = xr.DataArray(ccf, dims=('trials', 'roi', 'times'), name='CCF',
                       coords=(trials, roi_p, times_n))

    # add the windows used in the attributes
    ccf.attrs = check_attrs({**dict(type='ccf', normalized=normalized,
                                    times_unit=unit), **attrs})

    return ccf


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_trials = 20
    n_roi = 3
    n_times = 1000
    # create coordinates
    trials = np.arange(n_trials)
    roi = [f"roi_{k}" for k in range(n_roi)]
    times = (np.arange(n_times) - 200) / 64.
    # data creation
    x = np.random.rand(n_trials, n_roi, n_times)
    # inject relation
    bump = np.hanning(200).reshape(1, -1)
    x[:, 0, 200:400] += bump
    x[:, 1, 220:420] += bump
    x[:, 2, 260:460] += bump
    # xarray conversion
    x = xr.DataArray(x, dims=('trials', 'roi', 'times'),
                     coords=(trials, roi, times))
    plt.figure(figsize=(15, 6))

    # compute delayed dfc
    ccf = conn_ccf(x, times='times', roi='roi', n_jobs=1, verbose=False,
                   times_as_sample=True)
    print(ccf)

    plt.subplot(121)
    x.mean('trials').plot(x='times', hue='roi')
    plt.subplot(122)
    ccf.mean('trials').plot(x='times', hue='roi')
    plt.show()
