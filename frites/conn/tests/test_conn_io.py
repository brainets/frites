"""Test input conversion for connectivity functions."""
import numpy as np
import xarray as xr
import mne

from frites.conn import conn_io as cio, define_windows
from frites.utils.perf import id as id_arr


# generic data
n_trials, n_roi, n_times = 10, 7, 100
trials = np.array([0] * 5 + [1] * 5)
sfreq = 64.
roi_u = np.array([f"r{r}" for r in range(n_roi)])
roi = np.array(['r0', 'r0', 'R1', 'R1', 'R1', 'r2', 'r2'])
times = np.arange(n_times) / sfreq
data = np.random.rand(n_trials, n_roi, n_times)
data_xr = xr.DataArray(data, dims=('tr', 'space', 'ti'),
                       coords=(trials, roi, times))
info = mne.create_info(roi_u.tolist(), sfreq, ch_types='seeg')
data_mne = mne.EpochsArray(data, info, tmin=times[0], verbose=False)


def conn_io(*args, **kwargs):
    kwargs['verbose'] = 'error'
    return cio(*args, **kwargs)


class TestConnIO(object):

    @staticmethod
    def _test_memory(x, y):
        """Test that no internal copy have been made."""
        assert id_arr(x) == id_arr(y)

    def _test_data_out(self, x, data, roi=None, times=None, trials=None):
        self._test_memory(data, x.data)
        if roi is not None:
            np.testing.assert_array_equal(roi, x['roi'].data)
        if times is not None:
            np.testing.assert_array_equal(times, x['times'].data)
            assert x.attrs['sfreq'] == 64.
        if trials is not None:
            np.testing.assert_array_equal(trials, x['y'].data)

    def test_input_types(self):
        """Test input types."""
        # numpy types
        self._test_data_out(conn_io(data)[0], data)
        self._test_data_out(conn_io(data, roi=roi)[0], data, roi=roi)
        self._test_data_out(conn_io(data, times=times)[0], data, times=times)
        _x = conn_io(data, roi=roi, times=times)[0]
        self._test_data_out(_x, data, times=times, roi=roi)

        # xarray types
        _x = conn_io(data_xr, roi='space', times='ti')[0]
        self._test_data_out(_x, data, times=times, roi=roi, trials=trials)

        # mne types
        _x = conn_io(data_mne)[0]
        self._test_data_out(_x, data, times=times, roi=roi_u)

    def test_space_definition(self):
        """Test pairwise definitions."""
        # standard pairs name
        _, cfg = conn_io(data_xr, roi='space', times='ti', agg_ch=False)
        x_s, x_t = np.triu_indices(n_roi, k=1)
        roi_p = [f"{s}-{t}" for s, t in zip(roi[x_s], roi[x_t])]
        np.testing.assert_array_equal(cfg['roi_idx'].ravel(), np.arange(n_roi))
        np.testing.assert_array_equal(cfg['x_s'], x_s)
        np.testing.assert_array_equal(cfg['x_t'], x_t)
        np.testing.assert_array_equal(
            np.char.lower(cfg['roi_p']), np.char.lower(roi_p))

        # group of roi
        _, cfg = conn_io(data_xr, roi='space', times='ti', agg_ch=True)
        roi_p = ['r0-R1', 'r0-r2', 'R1-r2']
        roi_idx = [[0, 1], [2, 3, 4], [5, 6]]
        x_s, x_t = [0, 0, 1], [1, 2, 2]
        np.testing.assert_array_equal(cfg['roi_p'], roi_p)
        assert np.all([k == i for k, i in zip(cfg['roi_idx'], roi_idx)])
        np.testing.assert_array_equal(cfg['x_s'], x_s)
        np.testing.assert_array_equal(cfg['x_t'], x_t)

        # custom pairs of roi
        pairs = np.array([[0, 2], [4, 6]])
        _, cfg = conn_io(data_xr, roi='space', times='ti',
                         kw_links=dict(pairs=pairs))
        np.testing.assert_array_equal(cfg['x_s'], [0, 4])
        np.testing.assert_array_equal(cfg['x_t'], [2, 6])
        np.testing.assert_array_equal(cfg['roi_p'], ['r0-R1', 'R1-r2'])

        # test roi sorting
        pairs = np.array([[3, 0], [5, 2]])
        kw = dict(kw_links=dict(pairs=pairs, sort=False), roi='space',
                  times='ti')
        _, cfg = conn_io(data_xr, **kw)
        np.testing.assert_array_equal(cfg['roi_p'], ['R1-r0', 'r2-R1'])
        kw = dict(kw_links=dict(pairs=pairs, sort=True), roi='space',
                  times='ti')
        _, cfg = conn_io(data_xr, **kw)
        np.testing.assert_array_equal(cfg['roi_p'], ['r0-R1', 'R1-r2'])

    def test_temporal_definition(self):
        """Test sliding window definition"""
        # no window
        _, cfg = conn_io(data_xr, roi='space', times='ti')
        np.testing.assert_array_equal(
            cfg['win_sample'].ravel(), [0, n_times - 1])
        np.testing.assert_array_equal(
            cfg['win_times'], np.mean([times[0], times[-1]]))

        # sliding window
        win, _ = define_windows(times, slwin_len=.1, slwin_step=.01)
        _, cfg = conn_io(data_xr, roi='space', times='ti', win_sample=win)
        np.testing.assert_array_equal(cfg['win_sample'], win)
        np.testing.assert_array_equal(
            cfg['win_times'], times[cfg['win_sample']].mean(1))

    def test_task_related(self):
        """Test passing a y variable."""
        # test default
        x, _ = conn_io(data_xr, roi='space', times='ti')
        np.testing.assert_array_equal(x['trials'].data, np.arange(n_trials))
        np.testing.assert_array_equal(x['y'].data, trials)

        # test passing a task-variable
        y = np.random.rand(n_trials)
        x, _ = conn_io(data_xr, roi='space', times='ti', y=y)
        np.testing.assert_array_equal(x['trials'].data, np.arange(n_trials))
        np.testing.assert_array_equal(x['y'].data, y)

    def test_block_size(self):
        """Test block size definition"""
        trials = np.arange(n_trials)
        # no blocks
        _, cfg = conn_io(data_xr, roi='space', times='ti', block_size=None)
        np.testing.assert_array_equal(cfg['blocks'][0], trials)

        # custom block size
        for n_b in [2, 5]:
            _, cfg = conn_io(data_xr, roi='space', times='ti', block_size=n_b)
            assert len(cfg['blocks']) == n_b
            np.testing.assert_array_equal(np.r_[tuple(cfg['blocks'])], trials)


if __name__ == '__main__':
    TestConnIO().test_space_definition()
