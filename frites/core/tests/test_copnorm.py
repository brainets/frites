"""Test copnorm functions."""
import numpy as np

from frites.core import (copnorm_1d, copnorm_cat_1d, copnorm_cat_nd,
                         copnorm_nd)


class TestCopnorm(object):  # noqa

    def test_copnorm_1d(self):
        """Test function copnorm_1d."""
        arr = np.random.randint(0, 10, (20,))
        copnorm_1d(arr)

    def test_copnorm_cat_1d(self):
        """Test function copnorm_cat_1d."""
        arr = np.random.randint(0, 10, (20,))
        y = np.array([0] * 10 + [1] * 10)
        arr_ref = arr.copy()
        arr_ref[:10] = copnorm_1d(arr[:10])
        arr_ref[10:] = copnorm_1d(arr[10:])
        arr_c = copnorm_cat_1d(arr, y)
        np.testing.assert_array_equal(arr_ref, arr_c)

    def test_copnorm_nd(self):
        """Test function copnorm_nd."""
        _arr = np.random.randint(0, 10, (20,))
        arr_v = np.c_[_arr, _arr]
        arr_h = arr_v.T
        cp_v = copnorm_nd(arr_v, axis=0)
        cp_h = copnorm_nd(arr_h, axis=1)
        assert (cp_v[:, 0] == cp_v[:, 1]).all()
        assert (cp_h == cp_v.T).all()

    def test_copnorm_cat_nd(self):
        """Test function copnorm_cat_nd."""
        _arr = np.random.randint(0, 10, (20,))
        y = np.array([0] * 10 + [1] * 10)
        arr_v = np.c_[_arr, _arr]
        # manual copnorm
        arr_ref = arr_v.copy()
        arr_ref[:10, 0] = copnorm_1d(arr_v[:10, 0])
        arr_ref[10:, 0] = copnorm_1d(arr_v[10:, 0])
        arr_ref[:10, 1] = copnorm_1d(arr_v[:10, 1])
        arr_ref[10:, 1] = copnorm_1d(arr_v[10:, 1])
        # auto copnorm
        arr_c = copnorm_cat_nd(arr_v, y, axis=0)
        np.testing.assert_array_equal(arr_ref, arr_c)
