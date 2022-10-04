import numpy as np
import xarray as xr
import pandas as pd

from frites.conn import conn_reshape_undirected


def _prepare_plot_conn(
        conn, cmap=None, bad=None, vmin=None, vmax=None, categories=None,
        ax=None
    ):
    """Prepare inputs."""
    import matplotlib.pyplot as plt

    cfg = dict()

    # __________________________________ I/O __________________________________
    # connectivity array
    if isinstance(conn, np.ndarray):
        assert conn.ndim == 2
        n_rows, n_cols = conn.shape
        roi = np.arange(conn.shape[0]).astype(str)
        conn = pd.DataFrame(conn, index=np.arange(n_rows),
                            columns=np.arange(n_cols))
    elif isinstance(conn, xr.DataArray):
        assert conn.ndim == 2
        conn = conn.to_pandas()
    assert isinstance(conn, pd.DataFrame)
    np.testing.assert_array_equal(conn.index, conn.columns)
    conn.index = conn.columns = [str(k) for k in conn.index]

    # _________________________________ NODES _________________________________
    # _________________________________ COLOR _________________________________
    # colormap
    cmap = plt.get_cmap(cmap).copy()
    if bad:
        cmap.set_bad(color=bad)
    cfg['cmap'] = cmap

    # vmin, vmax trick
    if isinstance(vmin, str):
        vmin = np.nanpercentile(conn.values, float(vmin))
    if isinstance(vmax, str):
        vmax = np.nanpercentile(conn.values, float(vmax))
    cfg['vmin'], cfg['vmax'] = vmin, vmax

    # ______________________________ CATEGORIES _______________________________
    cfg['has_categories'] = False
    if isinstance(categories, (list, np.ndarray, tuple)):
        cat_cut = np.diff(np.unique(categories, return_inverse=True)[1]) != 0
        cut_at = np.where(cat_cut)[0] + 1
        cfg['has_categories'] = True
        cfg['categories'] = categories
        cfg['cut_at'] = cut_at.astype(int)

    # ________________________________ FIGURE _________________________________
    if ax is None:
        cfg['fig'] = plt.figure(figsize=(14, 6))
        cfg['ax'] = plt.gca()
    else:
        cfg['fig'] = plt.gcf()
        cfg['ax'] = plt.gca()
        plt.sca(ax)

    return conn, cfg


###############################################################################
###############################################################################
#                                 HEATMAP
###############################################################################
###############################################################################


def plot_conn_heatmap(
        conn, cmap='plasma', vmin=None, vmax=None, categories=None,
        categories_kw={}, cbar=True, cbar_title=None, cbar_kw={},
        bad=None, xticklabels='auto', yticklabels='auto', square=True, ax=None
    ):
    """Plot the connectivity matrix as a heatmap.

    Parameters
    ----------
    conn : xarray.DataArray | pandas.DataFrame | numpy.ndarray
        Either a 2D xarray.DataArray or a pandas DataFrame or a 2D NumPy array
    cmap : str | 'plasma'
        Colormap name
    vmin, vmax : float | None
        Minimum and maximum of colorbar limits
    categories : array_like | None
        Category associated to each region name. Can be hemisphere name,
        lobe name or indices describing group of regions. By default, an
        horizontal and a vertical lines are going to be plotted as a separation
        between categories (see argument below for controlling the aesthetic)
    categories_kw : dict | {}
        Additional arguments to control the aesthetic of the categorical lines
        (e.g. categories_kw={'color': 'orange', 'lw': 4})
    cbar : bool | True
        Add the colorbar
    cbar_title : str | None
        Colorbar title
    cbar_kw : dict | {}
        Additional arguments for controlling the colorbar title (e.g.
        cbar_kw={'fontweight': 'bold', 'fontsize': 20})
    bad : str | None
        Color of bad values in the connectivity matrix (nan or non-finite
        values). By default, pad pixels are transparent.
    xticklabels, yticklabels : str | None
        Use 'auto' for the automatic settings of the x and y tick labels. Use
        an integer to decrease the number of ticks displayed. You can also
        disable the tick labels using False
    square : bool | True
        Make the axis square
    ax : matplotlib Axes | None
        Matplotlib axis (to add to a subplot for example)

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap
    """
    import matplotlib.pyplot as plt

    # _________________________________ I/O ___________________________________
    # prepare inputs
    conn, cfg = _prepare_plot_conn(
        conn, categories=categories, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
        bad=bad
    )

    # unwrap config
    cmap = cfg['cmap']
    vmin, vmax = cfg['vmin'], cfg['vmax']
    fig, ax = cfg['fig'], cfg['ax']

    # _______________________________ HEATMAPS ________________________________
    # main heatmap
    plt.pcolormesh(
        conn.columns, conn.index, conn.values, vmin=vmin, vmax=vmax, cmap=cmap
    )
    plt.xticks(rotation=90)
    ax.invert_yaxis()
    if square:
        ax.set_aspect(1.)

    # colorbar
    if cbar:
        cbar = plt.colorbar()
        if cbar_title:
            cbar.set_label(cbar_title, **cbar_kw)

    # ______________________________ CATEGORIES _______________________________
    if cfg['has_categories']:
        for c in cfg['cut_at']:
            plt.axvline(c - .5, **categories_kw)
            plt.axhline(c - .5, **categories_kw)

    # _______________________________ X/Y TICKS _______________________________
    if not xticklabels:
        ax.set_xticks([])
    elif isinstance(xticklabels, int):
        ax.set_xticks(np.arange(len(conn.columns))[::xticklabels])
        ax.set_xticklabels(conn.columns[::xticklabels])
    if not yticklabels:
        ax.set_yticks([])
    elif isinstance(yticklabels, int):
        ax.set_yticks(np.arange(len(conn.columns))[::yticklabels])
        ax.set_yticklabels(conn.index[::yticklabels])


###############################################################################
###############################################################################
#                                   CIRCLE
###############################################################################
###############################################################################


def plot_conn_circle():
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from frites import set_mpl_style
    set_mpl_style()

    conn = np.random.rand(10, 10)
    cat = [0] * 3 + [1] * 7

    plot_conn_heatmap(conn, categories=cat, cmap='plasma', cbar_title='Test')
    plt.show()

