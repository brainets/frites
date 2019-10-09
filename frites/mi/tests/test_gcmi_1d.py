"""Test 1d implementations of gcmi."""
import numpy as np

from frites.mi.gcmi_1d import (ent_1d_g, mi_1d_gg, gcmi_1d_cc, mi_model_1d_gd,
                               gcmi_model_1d_cd, mi_mixture_1d_gd,
                               gcmi_mixture_1d_cd, cmi_1d_ggg, gccmi_1d_ccc,
                               gccmi_1d_ccd)

class TestGcmi1d(object):  # noqa

    def test_ent_1d_g(self):
        """Test function ent_1d_g."""
        uni = np.random.uniform(0, 10, (20,))
        duo = np.random.uniform(0, 10, (2, 20))
        ent_uni, ent_duo = ent_1d_g(uni), ent_1d_g(duo)
        assert ent_uni < ent_duo

    def test_mi_1d_gg(self):
        """Test function mi_1d_gg."""
        x_g = np.random.normal(size=(1000,))
        y_g = np.random.normal(size=(1000,))
        mi_1d_gg(x_g, y_g)

    def test_gcmi_1d_cc(self):
        """Test function gcmi_1d_cc."""
        x_c = np.random.uniform(0, 50, size=(1000,))
        y_c = np.random.uniform(0, 50, size=(1000,))
        gcmi_1d_cc(x_c, y_c)

    def test_mi_model_1d_gd(self):
        """Test function mi_model_1d_gd."""
        x_g = np.random.normal(size=(1000,))
        y_d = np.array([0] * 500 + [1] * 500)
        mi_model_1d_gd(x_g, y_d)

    def test_gcmi_model_1d_cd(self):
        """Test function gcmi_model_1d_cd."""
        x_c = np.random.uniform(0, 50, size=(1000,))
        y_d = np.array([0] * 500 + [1] * 500)
        gcmi_model_1d_cd(x_c, y_d)

    def test_mi_mixture_1d_gd(self):
        """Test function mi_mixture_1d_gd."""
        x_g = np.random.normal(size=(1000,))
        y_d = np.array([0] * 500 + [1] * 500)
        mi_mixture_1d_gd(x_g, y_d)

    def test_gcmi_mixture_cd(self):
        """Test function gcmi_mixture_cd."""
        x_c = np.random.uniform(0, 50, size=(1000,))
        y_d = np.array([0] * 500 + [1] * 500)
        gcmi_mixture_1d_cd(x_c, y_d)

    def test_cmi_1d_ggg(self):
        """Test function cmi_1d_ggg."""
        x_g = np.random.normal(size=(1000,))
        y_g = np.random.normal(size=(1000,))
        z_g = np.random.normal(size=(1000,))
        cmi_1d_ggg(x_g, y_g, z_g)

    def test_gccmi_1d_ccc(self):
        """Test function gccmi_1d_ccc."""
        x_c = np.random.uniform(0, 50, (1000,))
        y_c = np.random.uniform(0, 50, (1000,))
        z_c = np.random.uniform(0, 50, (1000,))
        gccmi_1d_ccc(x_c, y_c, z_c)

    def test_gccmi_1d_ccd(self):
        """Test function gccmi_1d_ccd."""
        x_c = np.random.uniform(0, 50, (1000,))
        y_c = np.random.uniform(0, 50, (1000,))
        z_d = np.array([0] * 500 + [1] * 500)
        gccmi_1d_ccd(x_c, y_c, z_d)
