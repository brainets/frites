"""Test multi-dimentional implementations of gcmi."""
import numpy as np

from frites.core.gcmi_nd import (mi_nd_gg, mi_model_nd_gd, cmi_nd_ggg,
                                 gcmi_nd_cc, gcmi_model_nd_cd, gccmi_nd_ccnd,
                                 gccmi_model_nd_cdnd, gccmi_nd_ccc, ent_nd_g)


class TestGcmiNd(object):  # noqa

    def ent_nd_g(self):
        """Test function ent_nd_g."""
        x_g = np.random.normal(size=(10, 1000, 20))
        assert ent_nd_g(x_g, traxis=1).shape == (10, 20)
        assert ent_nd_g(x_g, traxis=1, mvaxis=0).shape == (20,)

    def test_mi_nd_gg(self):
        """Test function mi_nd_gg."""
        x_g = np.random.normal(size=(10, 1000, 20))
        y_g = np.random.normal(size=(10, 1000, 20))
        assert mi_nd_gg(x_g, y_g, traxis=1).shape == (10, 20)
        assert mi_nd_gg(x_g, y_g, traxis=1, mvaxis=0).shape == (20,)

    def test_mi_model_nd_gd(self):
        """Test function mi_model_nd_gd."""
        x_g = np.random.normal(size=(10, 1000, 20))
        y_d = np.array([0] * 500 + [1] * 500)
        assert mi_model_nd_gd(x_g, y_d, traxis=1).shape == (10, 20)
        assert mi_model_nd_gd(x_g, y_d, traxis=1, mvaxis=-1).shape == (10,)

    def test_cmi_nd_ggg(self):
        """Test function cmi_nd_ggg."""
        x_g = np.random.normal(size=(10, 1000, 20))
        y_g = np.random.normal(size=(10, 1000, 20))
        z_g = np.random.normal(size=(10, 1000, 20))
        assert cmi_nd_ggg(x_g, y_g, z_g, traxis=1).shape == (10, 20)
        assert cmi_nd_ggg(x_g, y_g, z_g, traxis=1, mvaxis=0).shape == (20,)

    def test_gcmi_nd_cc(self):
        """Test function gcmi_nd_cc."""
        x_c = np.random.uniform(0, 50, size=(10, 1000, 20))
        y_c = np.random.uniform(0, 50, size=(10, 1000, 20))
        assert gcmi_nd_cc(x_c, y_c, traxis=1).shape == (10, 20)
        assert gcmi_nd_cc(x_c, y_c, traxis=1, mvaxis=0).shape == (20,)

    def test_gcmi_model_nd_cd(self):
        """Test function gcmi_model_nd_cd."""
        x_c = np.random.uniform(0, 50, size=(10, 1000, 20))
        y_d = np.array([0] * 500 + [1] * 500)
        assert gcmi_model_nd_cd(x_c, y_d, traxis=1).shape == (10, 20)
        assert gcmi_model_nd_cd(x_c, y_d, traxis=1, mvaxis=-1).shape == (10,)

    def test_gccmi_nd_ccnd(self):
        """Test function gccmi_nd_ccnd."""
        x_c = np.random.uniform(0, 50, size=(10, 1000, 20))
        y_c = np.random.uniform(0, 50, size=(10, 1000, 20))
        z_d = np.array([0] * 500 + [1] * 500)
        assert gccmi_nd_ccnd(x_c, y_c, z_d, traxis=1).shape == (10, 20)
        assert gccmi_nd_ccnd(x_c, y_c, z_d, traxis=1, mvaxis=-1).shape == (10,)

    def test_gccmi_model_nd_cdnd(self):
        """Test function gccmi_model_nd_cdnd."""
        x_c = np.random.uniform(0, 50, size=(10, 1000, 20))
        y_d = np.array([0] * 500 + [1] * 500)
        z_d = np.array([2] * 500 + [7] * 500)
        assert gccmi_model_nd_cdnd(x_c, y_d, z_d, traxis=1).shape == (10, 20)
        assert gccmi_model_nd_cdnd(x_c, y_d, z_d,
                                   traxis=1, mvaxis=-1).shape == (10,)

    def test_gccmi_nd_ccc(self):
        """Test function gccmi_nd_ccc."""
        x_c = np.random.uniform(0, 50, size=(10, 1000, 20))
        y_c = np.random.uniform(0, 50, size=(10, 1000, 20))
        z_c = np.random.uniform(0, 50, size=(10, 1000, 20))
        assert gccmi_nd_ccc(x_c, y_c, z_c, traxis=1).shape == (10, 20)
        assert gccmi_nd_ccc(x_c, y_c, z_c, traxis=1, mvaxis=0).shape == (20,)

if __name__ == "__main__":
    TestGcmiNd().ent_nd_g()
