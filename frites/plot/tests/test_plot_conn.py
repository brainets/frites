"""Test plotting connectivity."""
import numpy as np
import xarray as xr
import pandas as pd

from frites.plot.plot_conn import (
    _prepare_plot_conn, plot_conn_heatmap, plot_conn_circle
)


class TestPlotConn(object):

    @staticmethod
    def _get_conn(astype='numpy'):
        conn = [
            [np.nan, 2, 1],
            [2, 2, 4],
            [-1, np.nan, 1]
        ]
        roi = ["roi_0", "roi_1", "roi_2"]
        conn = np.asarray(conn)

        if astype == 'numpy':
            return conn
        elif astype == 'pandas':
            return pd.DataFrame(conn, index=roi, columns=roi)
        elif astype == 'xarray':
            return xr.DataArray(conn, dims=('sources', 'targets'),
                                coords=(roi, roi))

    def test_prepare_inputs(self):
        """Test input transformation for connectivity plots."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        plt.figure()

        # test input types
        for types in ['numpy', 'pandas', 'xarray']:
            conn = self._get_conn(astype=types)
            out = _prepare_plot_conn(conn)[0]
            assert isinstance(out, pd.DataFrame)

        # test color settings of edges
        conn = self._get_conn(astype='pandas')
        out = _prepare_plot_conn(conn, cmap='Set1', bad='orange')[1]
        assert isinstance(out['cmap'], ListedColormap)
        assert (out['vmin'] == -1.) and (out['vmax'] == 4.)
        out = _prepare_plot_conn(conn, vmin='20', vmax='80')[1]
        assert (out['vmin'] == 1.) and (out['vmax'] == 2.)
        plt.close()

        # test categories
        conn = self._get_conn(astype='pandas')
        out = _prepare_plot_conn(conn, categories=[0, 0, 1])[1]
        np.testing.assert_array_equal(out['categories'], [0, 0, 1])
        plt.close()

        # test nodes data
        conn = self._get_conn(astype='pandas')
        out = _prepare_plot_conn(conn, nodes_data=None)[1]
        np.testing.assert_array_equal(out['nodes_data'], [1.] * 3)
        out = _prepare_plot_conn(conn, nodes_data='degree')[1]
        np.testing.assert_array_equal(out['nodes_data'], [0, 0, 1])
        out = _prepare_plot_conn(conn, nodes_data='mean')[1]
        np.testing.assert_array_equal(out['nodes_data'], [0, 1, 1])
        out = _prepare_plot_conn(conn, nodes_data='diagonal')[1]
        np.testing.assert_array_equal(out['nodes_data'], [np.nan, 1, 0])
        out = _prepare_plot_conn(conn, nodes_data='number')[1]
        np.testing.assert_array_equal(out['nodes_data'], [0, .5, 1.])
        out = _prepare_plot_conn(conn, nodes_data='oups')[1]
        np.testing.assert_array_equal(out['nodes_data'], [1] * 3)
        out = _prepare_plot_conn(conn, nodes_data=[3, 2, 1])[1]
        np.testing.assert_array_equal(out['nodes_data'], [1, .5, 0])
        plt.close()

        # test nodes coloring
        conn = self._get_conn(astype='pandas')
        out = _prepare_plot_conn(conn, nodes_cmap='Set1', nodes_bad='k')[1]
        plt.close()

        # test axis definition
        conn = self._get_conn(astype='pandas')
        _prepare_plot_conn(conn, ax=None)
        _prepare_plot_conn(conn, ax=122)
        _prepare_plot_conn(conn, ax=(1, 2, 2))
        _prepare_plot_conn(conn, ax=plt.gca())
        plt.close()

        # test for squared axis
        conn = self._get_conn(astype='pandas')
        _prepare_plot_conn(conn, square=True)
        _prepare_plot_conn(conn, square=False)
        plt.close()

        # test for polar axis
        conn = self._get_conn(astype='pandas')
        _prepare_plot_conn(conn, polar=True)
        _prepare_plot_conn(conn, polar=False)
        plt.close()

        # test for edges proportion
        conn = self._get_conn(astype='pandas')
        out = _prepare_plot_conn(conn, prop=5)[0]
        np.testing.assert_array_equal(
            out, [[np.nan] * 3, [np.nan, np.nan, 4.], [np.nan] * 3]
        )
        plt.close()

    def test_plot_heatmap(self):
        """Test heatmap representation of connectivity."""
        import matplotlib.pyplot as plt
        conn = self._get_conn(astype='pandas')
        plot_conn_heatmap(conn, cbar=True, cbar_title='Never use jet')
        plot_conn_heatmap(conn, cbar=False, cmap='jet')  # :(
        plot_conn_heatmap(conn, categories=[0, 0, 1])
        plot_conn_heatmap(conn, xticklabels=True)
        plot_conn_heatmap(conn, xticklabels=10)
        plot_conn_heatmap(conn, xticklabels=False)
        plot_conn_heatmap(conn, yticklabels=True)
        plot_conn_heatmap(conn, yticklabels=10)
        plot_conn_heatmap(conn, yticklabels=False)
        plt.close()

    def test_plot_circular(self):
        """Test circular representation of connectivity."""
        import matplotlib.pyplot as plt
        conn = self._get_conn(astype='pandas')
        plot_conn_circle(conn, directed=False, edges_cmap='turbo')
        plot_conn_circle(conn, directed=True, edges_cmap='turbo')
        plot_conn_circle(conn, cbar=False)
        plot_conn_circle(conn, cbar=True, cbar_title='Use turbo instead')
        plt.close()


if __name__ == '__main__':
    TestPlotConn().test_plot_circular()
