"""Test copnorm functions."""
import numpy as np

from frites.mi import copnorm_1d, copnorm_nd  # noqa


class TestCopnorm(object):  # noqa

    def test_copnorm_1d(self):
        """Test function copnorm_1d."""
        arr = np.random.randint(0, 10, (20,))
        copnorm_1d(arr)

    def test_copnorm_nd(self):
        """Test function copnorm_nd."""
        _arr = np.random.randint(0, 10, (20,))
        arr_v = np.c_[_arr, _arr]
        arr_h = arr_v.T
        print(arr_h.shape, arr_v.shape)
        cp_v = copnorm_nd(arr_v, axis=0)
        cp_h = copnorm_nd(arr_h, axis=1)
        assert (cp_v[:, 0] == cp_v[:, 1]).all()
        assert (cp_h == cp_v.T).all()
