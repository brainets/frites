"""Test window functions."""
import numpy as np

from frites.utils import define_windows, plot_windows


class TestWindow(object):

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
