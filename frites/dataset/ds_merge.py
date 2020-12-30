"""Merge datasets."""
import numpy as np
from frites.dataset.ds_ephy import DatasetEphy


def merge_datasets(ds_list):
    """Merge electrophysiological datasets to build multivariate data.

    This function can be used to concatenate datasets in order to build
    multi-variate electrophysiological data.

    Parameters
    ----------
    ds_list : list
        List of datasets (:class:`frites.dataset.DatasetEphy`)

    Returns
    -------
    ds_cat : frites.dataset.DatasetEphy
        Concatenated datasets.
    """
    # inputs checking
    assert isinstance(ds_list, list)
    assert all([isinstance(k, DatasetEphy) for k in ds_list])
    n_datasets = len(ds_list)

    # group the data by roi
    for n_d in range(n_datasets):
        ds_list[n_d].groupby("roi")
    n_roi = len(ds_list[0]._x)
    assert all([len(k._x) == n_roi for k in ds_list])

    # concatenate inside the first dataset
    for n_r in range(n_roi):
        data_r = []
        for n_d in range(n_datasets):
            data_r.append(ds_list[n_d]._x[n_r])
        # put it inside the first dataset
        ds_list[0]._x[n_r] = np.concatenate(data_r, axis=-2)

    return ds_list[0]



if __name__ == '__main__':
    import xarray as xr

    n_datasets = 3
    n_subjects = 4
    n_roi = 10
    n_trials = 40
    n_times = 100
    roi = ["roi_0"] * 5 + ["roi_1"] * 5
    trials = np.arange(n_trials)
    times = np.arange(n_times) / 256.

    datasets = []
    for n_d in range(n_datasets):
        ds_suj = []
        for n_s in range(n_subjects):
            if n_d == 0:
                d = np.random.rand(n_trials, n_roi, n_times)
            elif n_d == 1:
                d = np.zeros((n_trials, n_roi, n_times))
            elif n_d == 2:
                d = np.ones((n_trials, n_roi, n_times))
            d = xr.DataArray(d, dims=('trials', 'roi', 'times'),
                             coords=(trials, roi, times))
            ds_suj += [d]
        datasets += [ds_suj]

    kw_ds = dict(y='trials', times='times', roi='roi')
    ds_list = [DatasetEphy(datasets[k], **kw_ds) for k in range(n_datasets)]
    ds_cat = merge_datasets(ds_list)

    import matplotlib.pyplot as plt
    plt.pcolormesh(ds_cat._x[1].mean(0))
    plt.show()
    print([k.shape for k in ds_cat._x])

