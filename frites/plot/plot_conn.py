"""Plot connectivity matrix.
"""
import numpy as np
import xarray as xr
import pandas as pd

from frites.utils import normalize


def _prepare_plot_conn(
        conn, cmap=None, bad=None, vmin=None, vmax=None, categories=None,
        nodes_data=None, nodes_cmap=None, nodes_bad=None, ax=None,
        square=False, prop=None, polar=False):
    """Prepare inputs."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    cfg = dict()

    # __________________________________ I/O __________________________________
    # connectivity array
    if isinstance(conn, np.ndarray):
        assert conn.ndim == 2
        n_rows, n_cols = conn.shape
        conn = pd.DataFrame(conn.copy(), index=np.arange(n_rows),
                            columns=np.arange(n_cols))
    elif isinstance(conn, xr.DataArray):
        assert conn.ndim == 2
        conn = conn.to_pandas().copy()
    else:
        conn = conn.copy()
    assert isinstance(conn, pd.DataFrame)
    np.testing.assert_array_equal(conn.index, conn.columns)
    conn.index = conn.columns = [str(k) for k in conn.index]
    n_nodes = len(conn.index)

    # proportion of edges to keep
    if isinstance(prop, (int, float)):
        assert 0 < prop < 100
        th = np.nanpercentile(conn.values, 100. - prop)
        conn[conn < th] = np.nan

    # _________________________________ NODES _________________________________
    # nodes name and order
    cfg['nodes_names'] = conn.index.tolist()
    cfg['nodes_order'] = conn.index.tolist()

    # nodes data
    if isinstance(nodes_data, (list, np.ndarray, tuple)):
        if len(nodes_data) != n_nodes:
            raise ValueError(
                "When passing custom data to the nodes, the `nodes_data` must "
                "have the same length as the number of nodes")
        nodes_data = np.asarray(nodes_data)
    elif nodes_data == 'degree':
        nodes_data = (np.isfinite(conn.values)).sum(axis=0).astype(int)
    elif nodes_data == 'mean':
        nodes_data = np.nanmean(conn.values, axis=0)
    elif nodes_data == 'diagonal':
        nodes_data = np.diag(conn.values)
    elif nodes_data == 'number':
        nodes_data = np.arange(n_nodes)
    else:
        nodes_data = np.full((n_nodes,), 1.)
    nodes_data = normalize(nodes_data, to_min=0., to_max=1.)
    cfg['nodes_data'] = np.ma.masked_array(
        nodes_data, mask=~np.isfinite(nodes_data)
    )

    # nodes color
    if (nodes_cmap is None) or isinstance(nodes_cmap, str):
        nodes_cmap = plt.get_cmap(nodes_cmap).copy()
    if nodes_bad:
        nodes_cmap.set_bad(color=nodes_bad)
    cfg['nodes_color'] = [nodes_cmap(k) for k in cfg['nodes_data']]

    # _________________________________ COLOR _________________________________
    # colormap
    if (cmap is None) or isinstance(cmap, str):
        cmap = plt.get_cmap(cmap).copy()
    if bad:
        cmap.set_bad(color=bad)
    cfg['cmap'] = cmap

    # vmin, vmax trick
    if isinstance(vmin, str):
        vmin = np.nanpercentile(conn.values, float(vmin))
    if isinstance(vmax, str):
        vmax = np.nanpercentile(conn.values, float(vmax))
    if not isinstance(vmin, (int, float)):
        vmin = np.nanmin(conn.values)
    if not isinstance(vmax, (int, float)):
        vmax = np.nanmax(conn.values)
    cfg['vmin'], cfg['vmax'] = float(vmin), float(vmax)

    # ______________________________ CATEGORIES _______________________________
    cfg['has_categories'] = False
    if isinstance(categories, (list, np.ndarray, tuple)):
        cat_cut = np.diff(np.unique(categories, return_inverse=True)[1]) != 0
        cut_at = np.where(cat_cut)[0] + 1
        cfg['has_categories'] = True
    else:
        cut_at = None
    cfg['categories'] = categories
    cfg['cut_at'] = cut_at

    # ________________________________ FIGURE _________________________________
    # check for existing axis and figure
    if isinstance(ax, (int, tuple, list, type(None), np.ndarray)):
        # deal with empty axis
        if ax is None:
            ax = 111
        if isinstance(ax, int):
            # ax = (ax,)
            ax = [int(k) for k in str(ax)]
        ax = tuple(ax)

        if all([k == 1 for k in ax]):
            cfg['fig'] = plt.figure(figsize=(14, 10))
        else:
            cfg['fig'] = plt.gcf()

        # create the axis
        cfg['fig'].add_subplot(*ax, polar=polar)
        cfg['ax'] = plt.gca()
    else:
        cfg['fig'] = plt.gcf()
        cfg['ax'] = ax
    plt.sca(cfg['ax'])

    # check for existing polar axis
    if polar and not isinstance(cfg['ax'], mpl.projections.polar.PolarAxes):
        raise ValueError("For circular plotting, axis should have a polar "
                         "projection (e.g. `plt.subplot(111, polar=True)`)")

    # square the axis (if needed)
    if square:
        cfg['ax'].set_aspect(1.)

    return conn, cfg


###############################################################################
###############################################################################
#                                 HEATMAP
###############################################################################
###############################################################################


def plot_conn_heatmap(
        conn, cmap='plasma', vmin=None, vmax=None, categories=None,
        categories_kw={}, cbar=True, cbar_title=None, cbar_kw={}, prop=None,
        bad=None, xticklabels='auto', yticklabels='auto', square=True,
        ax=None):
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
    prop : float | None
        Proportion of edges to keep. For example, prop=5 means that 95th
        percentile will be used for thresholding the connections.
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
        conn, categories=categories, vmin=vmin, vmax=vmax, cmap=cmap,
        bad=bad, prop=prop, ax=ax, square=square, polar=False
    )
    ax = cfg['ax']

    # _______________________________ HEATMAPS ________________________________
    # main heatmap
    plt.pcolormesh(
        conn.columns, conn.index, conn.values, vmin=cfg['vmin'],
        vmax=cfg['vmax'], cmap=cfg['cmap']
    )
    plt.xticks(rotation=90)
    ax.invert_yaxis()

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
        ax.set_yticks(np.arange(len(conn.index))[::yticklabels])
        ax.set_yticklabels(conn.index[::yticklabels])

    return ax


###############################################################################
###############################################################################
#                                   CIRCLE
###############################################################################
###############################################################################


def plot_conn_circle(
        conn, directed=False, edges_cmap='hot_r', edges_vmin=None,
        edges_vmax=None, edges_lw=3., edges_alpha=1., nodes_data='degree',
        nodes_cmap='hot_r', nodes_bad=None, nodes_fz=8, categories=None,
        categories_sep=3, cbar=True, cbar_title=None, cbar_kw={}, cbar_size=.8,
        cbar_pos=(.8, .4), prop=None, angle_start=90, angle_span=360,
        padding=0., ax=None):
    """Plot the connectivity matrix in a circle.

    .. note::

        Note: This code is based on the circle graph example by Nicolas P.
        Rougier http://www.labri.fr/perso/nrougier/coding/ and has then be
        modified within MNE-Python (v0.14)

    Parameters
    ----------
    conn : xarray.DataArray | pandas.DataFrame | numpy.ndarray
        Either a 2D xarray.DataArray or a pandas DataFrame or a 2D NumPy array
    directed : bool | False
        Specify whether it is directed connectivity (True) or undirected
        connectivity (False, default)
    edges_cmap : str | 'hot_r'
        Colormap to use for coloring the connections.
    edges_vmin : float | None
        Minimum value for colormap of edges. If None, the minimum among finite
        values will be used instead.
    edges_vmax : float | None
        Maximum value for colormap of edges. If None, the maximum among finite
        values will be used instead.
    edges_lw : float | 3.
        Line width of the strongest edges.
    edges_alpha : float | 1.
        Minimum transparency level for plotting edges. If 1 (default), all of
        the edges are going to be opaque while if 0 weakest edges are going to
        be transparent
    nodes_data : array_like, srt | None
        Data to use for coloring the boxes of each node. Use either :

            * A array like of length (n_nodes,)
            * 'degree' for the degree (i.e. the number of connections) of each
              node
            * 'mean' for the mean connectivity of this node
            * 'diagonal' use the values on the diagonal of the connectivity
              matrix
            * 'number' for coloring according to the node number
    nodes_cmap : str | 'hot_r'
        Colormap to use for coloring the nodes.
    nodes_bad : str | None
        Color to use for bad nodes (i.e. nodes with non finite values in
        nodes_data). By default, those nodes are transparent.
    nodes_fz : float | 8
        Font size of nodes' labels.
    categories : array_like | None
        Category associated to each region name. Can be hemisphere name,
        lobe name or indices describing group of regions. By default, a space
        is introduced between categories (see categories_sep below for
        controlling the amount of space to use)
    categories_sep : float | 3
        Space size between categories.
    cbar : bool | True
        Add the colorbar
    cbar_title : str | None
        Colorbar title
    cbar_kw : dict | {}
        Additional arguments for controlling the colorbar title (e.g.
        cbar_kw={'fontweight': 'bold', 'fontsize': 20})
    cbar_size : float | .8
        Size of the colorbar
    cbar_pos : tuple | (.8, .4)
        (x, y) position of the colorbar.
    prop : float | None
        Proportion of edges to keep. For example, prop=5 means that 95th
        percentile will be used for thresholding the connections.
    angle_start : float | 90
        Angle at which to start the first node (in degree)
    angle_span : float | 360
        Angle spanning (in degree). By default, a full circle is used (360°).
        For half a circle, use 180°.
    padding : float | 0.
        Add some space arround plot.
    ax : matplotlib Axes | None
        Matplotlib axis (to add to a subplot for example). The axis must have
        a polar projection (e.g. fig.add_subplot(1, 2, 2, polar=True))

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the circular plot
    """
    # _________________________________ I/O ___________________________________
    # prepare inputs
    conn, cfg = _prepare_plot_conn(
        conn, categories=categories, ax=ax, vmin=edges_vmin, vmax=edges_vmax,
        cmap=edges_cmap, bad=None, nodes_data=nodes_data,
        nodes_cmap=nodes_cmap, nodes_bad=nodes_bad, prop=prop, square=True,
        polar=True
    )

    # ________________________________ ANGLES _________________________________
    # get nodes information
    nodes_names, nodes_order = cfg['nodes_names'], cfg['nodes_order']

    # build angles
    angles = _circular_layout(
        nodes_names, nodes_order, start_pos=angle_start, start_between=True,
        group_boundaries=cfg['cut_at'], group_sep=categories_sep,
        span=angle_span
    )

    # plot the connectivity
    ax = _draw_conn_circle(
        conn.values, nodes_names, angles, cfg['nodes_color'], nodes_fz,
        cfg['cmap'], cfg['vmin'], cfg['vmax'], edges_lw, edges_alpha, directed,
        cbar, cbar_title, cbar_kw, cbar_size, cbar_pos, cfg['ax'], padding
    )

    return ax


def _circular_layout(
        node_names, node_order, start_pos=90, start_between=True,
        group_boundaries=None, group_sep=10, span=360):
    """Create layout arranging nodes on a circle.

    Parameters
    ----------
    node_names : list of str
        Node names.
    node_order : list of str
        List with node names defining the order in which the nodes are
        arranged. Must have the elements as node_names but the order can be
        different. The nodes are arranged clockwise starting at "start_pos"
        degrees.
    start_pos : float | 90
        Angle in degrees that defines where the first node is plotted.
    start_between : bool | True
        If True, the layout starts with the position between the nodes.
        This is the same as adding "180. / len(node_names)" to start_pos.
    group_boundaries : None | array-like
        List of of boundaries between groups at which point a "group_sep"
        will be inserted. E.g. "[0, len(node_names) / 2]" will create two
        groups.
    group_sep : float | 10
        Group separation angle in degrees. See "group_boundaries".

    Returns
    -------
    node_angles : array, shape=(n_node_names,)
        Node angles in degrees.
    """
    n_nodes = len(node_names)

    if len(node_order) != n_nodes:
        raise ValueError('node_order has to be the same length as node_names')

    if group_boundaries is not None:
        boundaries = np.array(group_boundaries, dtype=int)
        if np.any(boundaries >= n_nodes) or np.any(boundaries < 0):
            raise ValueError('"group_boundaries" has to be between 0 and '
                             'n_nodes - 1.')
        if len(boundaries) > 1 and np.any(np.diff(boundaries) <= 0):
            raise ValueError('"group_boundaries" must have non-decreasing '
                             'values.')
        n_group_sep = len(group_boundaries)
    else:
        n_group_sep = 0
        boundaries = None

    # convert it to a list with indices
    node_order = [node_order.index(name) for name in node_names]
    node_order = np.array(node_order)
    if len(np.unique(node_order)) != n_nodes:
        raise ValueError('node_order has repeated entries')

    node_sep = (span - n_group_sep * group_sep) / n_nodes

    if start_between:
        start_pos += node_sep / 2

        if boundaries is not None and boundaries[0] == 0:
            # special case when a group separator is at the start
            start_pos += group_sep / 2
            boundaries = boundaries[1:] if n_group_sep > 1 else None

    node_angles = np.ones(n_nodes, dtype=float) * node_sep
    node_angles[0] = start_pos
    if boundaries is not None:
        node_angles[boundaries] += group_sep

    node_angles = np.cumsum(node_angles)[node_order]

    # convert it to radians
    node_angles = node_angles * np.pi / 180

    return node_angles


def _draw_conn_circle(
        con, node_names, node_angles, node_colors, nodes_fz, edges_cmap,
        edges_vmin, edges_vmax, edges_lw, edges_alpha, directed, cbar,
        cbar_title, cbar_kw, cbar_size, cbar_pos, ax, padding,
        node_linewidth=2.):
    """Visualize connectivity as a circular graph."""
    import matplotlib.pyplot as plt
    import matplotlib.path as m_path
    import matplotlib.patches as m_patches

    n_nodes = len(node_names)

    # _________________________________ CONN __________________________________
    # we use the lower-triangular part
    assert con.ndim == 2
    if not directed:
        _indices = np.tril_indices(n_nodes, -1)
    else:
        _indices = np.where(~np.eye(n_nodes, dtype=bool))

    # drop non-finite values
    sources, targets = [], []
    for s, t in zip(_indices[0], _indices[1]):
        if np.isfinite(con[s, t]):
            sources.append(s)
            targets.append(t)
    indices = tuple([np.asarray(sources), np.asarray(targets)])
    con = con[indices]

    # sort connections
    sort_idx = np.argsort(con)
    con = con[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # Get vmin vmax for color scaling
    vrange = edges_vmax - edges_vmin

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    con_val_scaled = (con - edges_vmin) / vrange

    # scale linewidth
    lw = normalize(con_val_scaled, 1., edges_lw)

    # scale transparency
    alphas = normalize(con_val_scaled, edges_alpha, 1.)

    # ________________________________ STYLE __________________________________
    # No ticks, we'll put our own
    plt.xticks([])
    plt.yticks([])

    # Set y axes limit, add additional space if requested
    plt.ylim(0, 10 + padding)

    # Remove the black axes border which may obscure the labels
    ax.spines['polar'].set_visible(False)

    # information for directed connectivity
    arrowstyle = '->,head_length=.6,head_width=.4'

    # ________________________________ NOISE __________________________________
    # widths correspond to the minimum angle between two nodes
    dist_mat = node_angles[None, :] - node_angles[:, None]
    dist_mat[np.diag_indices(n_nodes)] = 1e9
    node_width = np.min(np.abs(dist_mat))

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    nodes_n_con = np.zeros((n_nodes), dtype=int)
    for i, j in zip(indices[0], indices[1]):
        nodes_n_con[i] += 1
        nodes_n_con[j] += 1

    # initialize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(seed=0)

    n_con = len(indices[0])
    noise_max = 0.25 * node_width
    start_noise = rng.uniform(-noise_max, noise_max, n_con)
    end_noise = rng.uniform(-noise_max, noise_max, n_con)

    nodes_n_con_seen = np.zeros_like(nodes_n_con)
    for i, (start, end) in enumerate(zip(indices[0], indices[1])):
        nodes_n_con_seen[start] += 1
        nodes_n_con_seen[end] += 1

        start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start]) /
                           float(nodes_n_con[start]))
        end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end]) /
                         float(nodes_n_con[end]))

    # ________________________________ LINES __________________________________
    # Finally, we draw the connections
    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
        # Start point
        t0, r0 = node_angles[i], 10

        # End point
        if directed:
            # make shorter to accomodate arrowhead
            t1, r1 = node_angles[j], 9
        else:
            t1, r1 = node_angles[j], 10

        # Some noise in start and end point
        t0 += start_noise[pos]
        t1 += end_noise[pos]

        verts = [(t0, r0), (t0, 5), (t1, 5), (t1, r1)]
        codes = [m_path.Path.MOVETO, m_path.Path.CURVE4, m_path.Path.CURVE4,
                 m_path.Path.LINETO]
        path = m_path.Path(verts, codes)

        color = edges_cmap(con_val_scaled[pos])

        if directed:
            patch = m_patches.FancyArrowPatch(
                path=path, arrowstyle=arrowstyle, fill=False, edgecolor=color,
                mutation_scale=10, linewidth=lw[pos], alpha=alphas[pos]
            )
        else:
            patch = m_patches.PathPatch(
                path, fill=False, edgecolor=color, linewidth=lw[pos],
                alpha=alphas[pos]
            )

        ax.add_patch(patch)

    # ________________________________ BOXES __________________________________
    # Draw ring with colored nodes
    height = np.ones(n_nodes) * 1.
    bars = ax.bar(
        node_angles, height, width=node_width, bottom=9, edgecolor='w',
        lw=node_linewidth, facecolor='.9', align='center'
    )
    for bar, color in zip(bars, node_colors):
        bar.set_facecolor(color)

    # ________________________________ LABELS _______________________________
    angles_deg = 180 * node_angles / np.pi
    for name, angle_rad, angle_deg in zip(node_names, node_angles, angles_deg):
        if angle_deg >= 270:
            ha = 'left'
        else:
            # Flip the label, so text is always upright
            angle_deg += 180
            ha = 'right'

        ax.text(
            angle_rad, 10.2, name, size=nodes_fz, rotation=angle_deg,
            rotation_mode='anchor', horizontalalignment=ha,
            verticalalignment='center'
        )

    # ________________________________ COLORBAR _______________________________
    if cbar:
        sm = plt.cm.ScalarMappable(cmap=edges_cmap,
                                   norm=plt.Normalize(edges_vmin, edges_vmax))
        sm.set_array(np.linspace(edges_vmin, edges_vmax))
        cbar = plt.colorbar(
            sm, ax=ax, use_gridspec=False, shrink=cbar_size, anchor=cbar_pos
        )
        if cbar_title:
            cbar.set_label(cbar_title, **cbar_kw)

    return ax


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from frites import set_mpl_style
    set_mpl_style()

    # conn = np.random.rand(10, 10)
    conn = np.arange(100).reshape(10, 10)
    cat = [0] * 3 + [1] * 7

    # plot_conn_heatmap(conn, categories=cat, cmap='plasma', cbar_title='Test')
    plot_conn_circle(
        conn, categories=cat, edges_cmap='hot_r', cbar_title='Test',
        angle_span=180, categories_sep=20, nodes_data='diagonal',
        nodes_cmap='Spectral_r'
    )
    plt.show()
