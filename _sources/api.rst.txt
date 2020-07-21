API
===

.. contents::
   :local:
   :depth: 2


Dataset
-------

.. currentmodule:: frites.dataset

.. automodule:: frites.dataset
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   DatasetEphy

.. raw:: html

  <hr>


Workflow
--------

.. currentmodule:: frites.workflow

.. automodule:: frites.workflow
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   WfMi
   WfConn
   WfFit
   WfStatsEphy

.. raw:: html

  <hr>

Connectivity
------------

.. currentmodule:: frites.conn

.. automodule:: frites.conn
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   conn_dfc
   conn_covgc
   conn_fit
   conn_transfer_entropy


.. raw:: html

  <hr>

Statistics
----------

.. currentmodule:: frites.stats

.. automodule:: frites.stats
   :no-members:
   :no-inherited-members:

Correction for multiple comparisons
+++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   cluster_correction_mcp
   testwise_correction_mcp
   cluster_threshold


Random effect (rfx)
+++++++++++++++++++

.. autosummary::
   :toctree: generated/

   ttest_1samp
   rfx_ttest

.. raw:: html

  <hr>

I/O
---

.. currentmodule:: frites.io

.. automodule:: frites.io
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   convert_spatiotemporal_outputs
   convert_dfc_outputs

.. raw:: html

  <hr>

Simulations
-----------


.. currentmodule:: frites.simulations

.. automodule:: frites.simulations
   :no-members:
   :no-inherited-members:

Simulate electrophysiological data
++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   sim_single_suj_ephy
   sim_multi_suj_ephy

Simulate data for testing mutual information
++++++++++++++++++++++++++++++++++++++++++++

From random data
****************

.. autosummary::
   :toctree: generated/

   sim_local_cc_ss
   sim_local_cc_ms
   sim_local_cd_ss
   sim_local_cd_ms
   sim_local_ccd_ms
   sim_local_ccd_ss

From real data
****************

.. autosummary::
   :toctree: generated/

   sim_mi_cc
   sim_mi_cd
   sim_mi_ccd

.. raw:: html

  <hr>

Simulate data for testing connectivity
++++++++++++++++++++++++++++++++++++++

Random data for directed connectivity measures
**********************************************

.. autosummary::
   :toctree: generated/

   sim_distant_cc_ms
   sim_distant_cc_ss
   sim_gauss_fit

.. raw:: html

  <hr>

Utils
-----

.. currentmodule:: frites.utils

.. automodule:: frites.utils
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   define_windows
   plot_windows

.. raw:: html

  <hr>

Configuration
-------------

.. currentmodule:: frites

.. autosummary::
   :toctree: generated/

   get_config
   set_config

.. raw:: html

  <hr>

Core
----

.. currentmodule:: frites.core

.. automodule:: frites.core
   :no-members:
   :no-inherited-members:


Gaussian-Copula (1d)
++++++++++++++++++++

Gaussian-Copula based measures to apply to unidimensional vectors

.. autosummary::
   :toctree: generated/

   copnorm_1d
   copnorm_cat_1d
   ent_1d_g
   mi_1d_gg
   mi_model_1d_gd
   mi_mixture_1d_gd
   cmi_1d_ggg
   gcmi_1d_cc
   gcmi_model_1d_cd
   gcmi_mixture_1d_cd
   gccmi_1d_ccc
   gccmi_1d_ccd

Gaussian-copula (Nd)
++++++++++++++++++++

Gaussian-Copula based measures to apply to multidimensional vectors

.. autosummary::
   :toctree: generated/

   copnorm_nd
   copnorm_cat_nd
   mi_nd_gg
   mi_model_nd_gd
   cmi_nd_ggg
   gcmi_nd_cc
   gcmi_model_nd_cd
   gccmi_nd_ccnd
   gccmi_model_nd_cdnd
   gccmi_nd_ccc
