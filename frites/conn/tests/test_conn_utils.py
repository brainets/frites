"""Test window functions."""
import numpy as np
import xarray as xr

from frites.conn import (conn_reshape_undirected, conn_reshape_directed,
                         conn_ravel_directed, define_windows, plot_windows,
                         conn_dfc, conn_covgc, conn_get_pairs, conn_net,
                         conn_links)


class TestConnUtils(object):

    def test_conn_reshape_undirected(self):
        """Test function conn_reshape_undirected."""
        import pandas as pd
        # compute DFC
        n_epochs, n_times, n_roi = 5, 100, 3
        times = np.linspace(-1, 1, n_times)
        win_sample = np.array([[10, 20], [30, 40]])
        roi = [f"roi_{k}" for k in range(n_roi)]
        order = ['roi_2', 'roi_1']
        x = np.random.rand(n_epochs, n_roi, n_times)
        dfc = conn_dfc(x, win_sample, times=times, roi=roi).mean('trials')
        # reshape it without the time dimension
        dfc_mean = conn_reshape_undirected(dfc.mean('times'))
        assert dfc_mean.shape == (n_roi, n_roi, 1)
        df = conn_reshape_undirected(dfc.mean('times'), order=order,
                                     to_dataframe=True)
        assert isinstance(df, pd.DataFrame)
        # reshape it with the time dimension
        dfc_times = conn_reshape_undirected(dfc.copy())
        assert dfc_times.shape == (n_roi, n_roi, len(dfc['times']))
        # try the reorder
        dfc_order = conn_reshape_undirected(dfc, order=order)
        assert dfc_order.shape == (2, 2, len(dfc['times']))
        assert np.array_equal(dfc_order['sources'], dfc_order['targets'])
        assert np.array_equal(dfc_order['sources'], order)

    def test_conn_reshape_directed(self):
        """Test function conn_reshape_directed."""
        n_epochs, n_times, n_roi = 5, 100, 3
        x = np.random.rand(n_epochs, n_roi, n_times)
        dt, lag, t0 = 10, 2, [50, 80]
        order = ['roi_2', 'roi_1']
        # compute covgc
        gc = conn_covgc(x, dt, lag, t0, n_jobs=1, method='gauss')
        gc = gc.mean('trials')
        # reshape it without the time dimension
        gc_mean = conn_reshape_directed(gc.copy().mean('times'))
        assert gc_mean.shape == (n_roi, n_roi, 1)
        # reshape it with the time dimension
        gc_times = conn_reshape_directed(gc.copy())
        assert gc_times.shape == (n_roi, n_roi, len(gc['times']))
        # try the reorder
        gc_order = conn_reshape_directed(gc.copy(), order=order)
        assert gc_order.shape == (2, 2, len(gc['times']))
        assert np.array_equal(gc_order['sources'], gc_order['targets'])
        assert np.array_equal(gc_order['sources'], order)

    def test_define_windows(self):
        """Test function define_windows."""
        n_pts = 1000
        times = np.linspace(-1, 1, n_pts, endpoint=True)
        kw = dict(verbose=False)

        # ---------------------------------------------------------------------
        # full window
        # ---------------------------------------------------------------------
        ts = define_windows(times, **kw)[0]
        np.testing.assert_array_equal(ts, np.array([[0, n_pts - 1]]))

        # ---------------------------------------------------------------------
        # custom windows
        # ---------------------------------------------------------------------
        # single window
        win = [0, .5]
        ts = define_windows(times, windows=win, **kw)[0]
        np.testing.assert_almost_equal(times[ts.ravel()], win, decimal=2)
        # multiple
        win = [[-.5, -.1], [-.1, 0.], [0, .5]]
        ts = define_windows(times, windows=win, **kw)[0]
        np.testing.assert_almost_equal(times[ts], np.array(win), decimal=2)

        # ---------------------------------------------------------------------
        # sliding windows
        # ---------------------------------------------------------------------
        # length only
        ts = define_windows(times, slwin_len=.1, **kw)[0]
        tts = times[ts]
        ttsd = np.diff(tts, axis=1)
        np.testing.assert_almost_equal(ttsd, np.full_like(ttsd, .1), decimal=2)
        # with starting point
        ts = define_windows(times, slwin_len=.1, slwin_start=.5, **kw)[0]
        np.testing.assert_almost_equal(times[ts][0, 0], .5, decimal=2)
        # with stoping point
        ts = define_windows(times, slwin_len=.1, slwin_stop=.9,
                            **kw)[0]
        assert times[ts][-1, -1] <= .9
        # with step between temporal windows
        ts = define_windows(times, slwin_len=.1, slwin_step=.2, **kw)[0]
        tts = times[ts]
        ttsd = np.diff(tts, axis=0)
        np.testing.assert_almost_equal(ttsd, np.full_like(ttsd, .2), decimal=2)

    def test_plot_windows(self):
        """Test function plot_windows."""
        n_pts = 1000
        times = np.linspace(-1, 1, n_pts, endpoint=True)
        kw = dict(verbose=False)
        ts = define_windows(times, slwin_len=.1, **kw)[0]
        plot_windows(times, ts)

    def test_conn_get_pairs(self):
        """Test function conn_get_pairs."""
        roi = [np.array(['r1', 'r0']), np.array(['r0', 'r2', 'r1'])]
        # test non-directed
        df, _ = conn_get_pairs(roi, directed=False)
        rundir = np.c_[['r0', 'r0', 'r1'], ['r1', 'r2', 'r2']]
        names = [f'{k}-{i}' for k, i in zip(rundir[:, 0], rundir[:, 1])]
        suj = [0, 1, 1, 1]
        nsuj = [2, 1, 1]
        assert np.all(df['keep'])
        np.testing.assert_array_equal(df['sources'], rundir[:, 0])
        np.testing.assert_array_equal(df['targets'], rundir[:, 1])
        np.testing.assert_array_equal(df['#subjects'], nsuj)
        np.testing.assert_array_equal(df['names'], names)
        np.testing.assert_array_equal(np.concatenate(df['subjects']), suj)
        # test directed
        df, _ = conn_get_pairs(roi, directed=True)
        rdir = np.c_[['r0', 'r0', 'r1', 'r1', 'r2', 'r2'],
                     ['r1', 'r2', 'r0', 'r2', 'r0', 'r1']]
        names = [f'{k}->{i}' for k, i in zip(rdir[:, 0], rdir[:, 1])]
        suj = [0, 1, 1, 0, 1, 1, 1, 1]
        nsuj = [2, 1, 2, 1, 1, 1]
        assert np.all(df['keep'])
        np.testing.assert_array_equal(df['sources'], rdir[:, 0])
        np.testing.assert_array_equal(df['targets'], rdir[:, 1])
        np.testing.assert_array_equal(df['#subjects'], nsuj)
        np.testing.assert_array_equal(df['names'], names)
        np.testing.assert_array_equal(np.concatenate(df['subjects']), suj)
        # test nb_min_suj filtering (non-directed)
        df, _ = conn_get_pairs(roi, directed=False, nb_min_suj=2)
        np.testing.assert_array_equal(df['keep'], [True, False, False])
        df = df.loc[df['keep']]
        np.testing.assert_array_equal(df['sources'], ['r0'])
        np.testing.assert_array_equal(df['targets'], ['r1'])
        np.testing.assert_array_equal(df['#subjects'], [2])
        np.testing.assert_array_equal(np.concatenate(df['subjects']), [0, 1])
        np.testing.assert_array_equal(df['names'], ['r0-r1'])
        # test nb_min_suj filtering (directed)
        df, _ = conn_get_pairs(roi, directed=True, nb_min_suj=2)
        np.testing.assert_array_equal(
            df['keep'], [True, False, True, False, False, False])
        df = df.loc[df['keep']]
        np.testing.assert_array_equal(df['sources'], ['r0', 'r1'])
        np.testing.assert_array_equal(df['targets'], ['r1', 'r0'])
        np.testing.assert_array_equal(df['#subjects'], [2, 2])
        np.testing.assert_array_equal(
            np.concatenate(list(df['subjects'])), [0, 1, 0, 1])
        np.testing.assert_array_equal(df['names'], ['r0->r1', 'r1->r0'])

    def test_conn_ravel_directed(self):
        """Test function conn_ravel_directed."""
        n_trials = 100
        n_times = 1000
        n_roi = 3
        x_s, x_t = np.triu_indices(n_roi, k=1)

        # build coordinates
        roi = [f'r{s}-r{t}' for s, t in zip(x_s, x_t)]
        trials = np.random.randint(0, 2, (n_trials,))
        times = np.arange(n_times) / 64.
        direction = ['x->y', 'y->x']
        roi_dir = ['r0->r1', 'r0->r2', 'r1->r2', 'r1->r0', 'r2->r0', 'r2->r1']

        # create the connectivity arrays
        conn_xy = np.random.rand(n_trials, n_roi, n_times)
        conn_yx = np.random.rand(n_trials, n_roi, n_times)
        conn_c = np.concatenate((conn_xy, conn_yx), axis=1)

        # stack them and xarray conversion
        conn = np.stack((conn_xy, conn_yx), axis=-1)
        conn = xr.DataArray(conn, dims=('trials', 'roi', 'times', 'direction'),
                            coords=(trials, roi, times, direction))

        # ravel the array
        conn_r = conn_ravel_directed(conn)
        assert len(conn_r.shape) == 3
        np.testing.assert_array_equal(conn_r['roi'].data, roi_dir)
        np.testing.assert_array_equal(conn_r.data, conn_c)

    def test_conn_net(self):
        """Test function conn_net."""
        conn_xy = np.full((10, 3, 12), 2)
        conn_yx = np.ones((10, 4, 12))
        conn = np.concatenate((conn_xy, conn_yx), axis=1)
        roi = ['x->y', 'x->z', 'y->z', 'y->x', 'z->x', 'z->y', 'z->a']
        trials, times = np.arange(10), np.arange(12)
        conn = xr.DataArray(conn, dims=('trials', 'space', 'times'),
                            coords=(trials, roi, times))

        # test normal usage
        net = conn_net(conn, roi='space', sep='->', invert=False)
        np.testing.assert_array_equal(net.shape, (10, 3, 12))
        np.testing.assert_array_equal(net['trials'], trials)
        np.testing.assert_array_equal(net['times'], times)
        np.testing.assert_array_equal(net['space'], ['x-y', 'x-z', 'y-z'])
        np.testing.assert_array_equal(
            net.attrs['net_source'], ['x->y', 'x->z', 'y->z'])
        np.testing.assert_array_equal(
            net.attrs['net_target'], ['y->x', 'z->x', 'z->y'])
        np.testing.assert_array_equal(net.data, np.full((10, 3, 12), 1))

        # test inverted
        net = conn_net(conn, roi='space', sep='->', invert=True)
        np.testing.assert_array_equal(net['space'], ['y-x', 'z-x', 'z-y'])
        np.testing.assert_array_equal(
            net.attrs['net_source'], ['y->x', 'z->x', 'z->y'])
        np.testing.assert_array_equal(
            net.attrs['net_target'], ['x->y', 'x->z', 'y->z'])
        np.testing.assert_array_equal(net.data, np.full((10, 3, 12), -1))

        # test order
        net = conn_net(conn, roi='space', sep='->', order=['z', 'x', 'y'])
        np.testing.assert_array_equal(net['space'], ['z-x', 'z-y', 'x-y'])

    def test_conn_links(self):
        """Test function conn_links."""
        roi = ['dlPFC', 'aINS', 'dlPFC', 'vmPFC']
        # overall testing
        (x_s, x_t), roi_st = conn_links(roi, verbose=False)
        assert len(x_s) == len(x_t) == len(roi_st)
        assert all([f"{roi_st[s]}-{roi_st[t]}" for s, t in zip(x_s, x_t)])

        # test direction
        _, roi_st = conn_links(roi)
        np.testing.assert_array_equal(
            roi_st, [
                'aINS-dlPFC', 'dlPFC-dlPFC', 'dlPFC-vmPFC', 'aINS-dlPFC',
                'aINS-vmPFC', 'dlPFC-vmPFC'])
        _, roi_st = conn_links(roi, directed=True, net=False)
        np.testing.assert_array_equal(
            roi_st, [
                'dlPFC->aINS', 'dlPFC->dlPFC', 'dlPFC->vmPFC', 'aINS->dlPFC',
                'aINS->dlPFC', 'aINS->vmPFC', 'dlPFC->dlPFC', 'dlPFC->aINS',
                'dlPFC->vmPFC', 'vmPFC->dlPFC', 'vmPFC->aINS', 'vmPFC->dlPFC'])
        _, roi_st = conn_links(roi, directed=True, net=True)
        np.testing.assert_array_equal(
            roi_st, ['aINS-dlPFC', 'dlPFC-dlPFC', 'dlPFC-vmPFC', 'aINS-dlPFC',
                     'aINS-vmPFC', 'dlPFC-vmPFC'])

        # testing removing intra / inter roi connections
        _, roi_st = conn_links(roi, roi_relation='intra')
        np.testing.assert_array_equal(roi_st, ['dlPFC-dlPFC'])
        _, roi_st = conn_links(roi, roi_relation='inter')
        np.testing.assert_array_equal(roi_st, [
            'aINS-dlPFC', 'dlPFC-vmPFC', 'aINS-dlPFC', 'aINS-vmPFC',
            'dlPFC-vmPFC'])
        _, roi_st = conn_links(roi, directed=True, net=False,
                               roi_relation='intra')
        np.testing.assert_array_equal(roi_st, ['dlPFC->dlPFC', 'dlPFC->dlPFC'])
        _, roi_st = conn_links(roi, directed=True, net=False,
                               roi_relation='inter')
        np.testing.assert_array_equal(roi_st, [
            'dlPFC->aINS', 'dlPFC->vmPFC', 'aINS->dlPFC', 'aINS->dlPFC',
            'aINS->vmPFC', 'dlPFC->aINS', 'dlPFC->vmPFC', 'vmPFC->dlPFC',
            'vmPFC->aINS', 'vmPFC->dlPFC'])

        # remove links without a minimum number of connections
        _, roi_st = conn_links(roi, nb_min_links=2)
        np.testing.assert_array_equal(
            roi_st, ['aINS-dlPFC', 'dlPFC-vmPFC', 'aINS-dlPFC', 'dlPFC-vmPFC'])
        _, roi_st = conn_links(roi, directed=True, nb_min_links=2)
        np.testing.assert_array_equal(
            roi_st, [
                'dlPFC->aINS', 'dlPFC->dlPFC', 'dlPFC->vmPFC', 'aINS->dlPFC',
                'aINS->dlPFC', 'dlPFC->dlPFC', 'dlPFC->aINS', 'dlPFC->vmPFC',
                'vmPFC->dlPFC', 'vmPFC->dlPFC'])
        _, roi_st = conn_links(roi, directed=True, nb_min_links=2, net=True)
        np.testing.assert_array_equal(
            roi_st, ['aINS-dlPFC', 'dlPFC-vmPFC', 'aINS-dlPFC', 'dlPFC-vmPFC'])

        # testing string separator
        for direction in [True, False]:
            _, roi_st = conn_links(roi, sep='/', directed=True)
            assert all(['/' in r for r in roi_st])

        # testing pairs
        p_1, p_2 = np.array([0, 2]), np.array([1, 3])
        _, roi_st = conn_links(roi, pairs=np.c_[p_1, p_2])
        np.testing.assert_array_equal(
            roi_st, ['aINS-dlPFC', 'dlPFC-vmPFC'])
        _, roi_st = conn_links(roi, pairs=np.c_[p_1, p_2], directed=True)
        np.testing.assert_array_equal(
            roi_st, ['dlPFC->aINS', 'dlPFC->vmPFC'])

        # test hemispheric selection
        hemi = ['R', 'R', 'L', 'L']
        roi_2 = ['r0', 'r1', 'r2', 'r3']
        _, roi_st = conn_links(roi_2, hemisphere=hemi, hemi_links='both')
        np.testing.assert_array_equal(
            roi_st, ['r0-r1', 'r0-r2', 'r0-r3', 'r1-r2', 'r1-r3', 'r2-r3'])
        _, roi_st = conn_links(roi_2, hemisphere=hemi, hemi_links='intra')
        np.testing.assert_array_equal(roi_st, ['r0-r1', 'r2-r3'])
        _, roi_st = conn_links(roi_2, hemisphere=hemi, hemi_links='inter')
        np.testing.assert_array_equal(
            roi_st, ['r0-r2', 'r0-r3', 'r1-r2', 'r1-r3'])

        # test seed selection
        roi_3 = ['r0', 'r0', 'r1', 'r2', 'r3']
        _, roi_st = conn_links(roi_3, source_seed='r2')
        np.testing.assert_array_equal(
            roi_st, ['r0-r2', 'r0-r2', 'r1-r2', 'r2-r3'])
        _, roi_st = conn_links(roi_3, source_seed='r2', directed=True)
        np.testing.assert_array_equal(
            roi_st, ['r2->r0', 'r2->r0', 'r2->r1', 'r2->r3'])
        _, roi_st = conn_links(roi_3, target_seed='r3')
        np.testing.assert_array_equal(
            roi_st, ['r0-r3', 'r0-r3', 'r1-r3', 'r2-r3'])
        _, roi_st = conn_links(roi_3, target_seed='r3', directed=True)
        np.testing.assert_array_equal(
            roi_st, ['r0->r3', 'r0->r3', 'r1->r3', 'r2->r3'])
        _, roi_st = conn_links(roi_3, source_seed=['r2', 'r3'], directed=True)
        np.testing.assert_array_equal(
            roi_st, [
                'r2->r0', 'r2->r0', 'r2->r1', 'r2->r3', 'r3->r0', 'r3->r0',
                'r3->r1', 'r3->r2'])
        _, roi_st = conn_links(roi_3, target_seed=['r2', 'r3'], directed=True)
        np.testing.assert_array_equal(
            roi_st, [
                'r0->r2', 'r0->r3', 'r0->r2', 'r0->r3', 'r1->r2', 'r1->r3',
                'r2->r3', 'r3->r2'])
        _, roi_st = conn_links(roi_3, source_seed='r2', target_seed='r3',
                               directed=True)
        np.testing.assert_array_equal(roi_st, ['r2->r3'])


if __name__ == '__main__':
    TestConnUtils().test_define_windows()
