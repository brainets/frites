"""Test performance tools."""
import numpy as np
from time import sleep

from frites.utils.perf import timeit, id, get_data_base, arrays_share_data


class TestPerfTools(object):

    def test_timeit(self):
        @timeit
        def fcn(sec): return sleep(sec)  # noqa
        fcn(1.)

    def test_id(self):
        x = np.random.rand(1000)
        assert id(x) != id(x.copy())
        assert id(x) == id(x)

    def test_get_data_base(self):
        x = np.random.rand(1000)
        np.testing.assert_array_almost_equal(x, get_data_base(x))

    def test_arrays_share_data(self):
        x = np.random.rand(1000)
        y = np.random.rand(1000)
        assert not arrays_share_data(x, y)
        assert arrays_share_data(x, x)
