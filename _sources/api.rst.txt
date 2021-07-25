API
===

.. contents::
   :local:
   :depth: 2

.. note::

    This first section contains the most-used functions and classes for a user

.. ----------------------------------------------------------------------------

Dataset
-------

:py:mod:`frites.dataset`:

.. currentmodule:: frites.dataset

.. automodule:: frites.dataset
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   SubjectEphy
   DatasetEphy

.. raw:: html

  <hr>

.. ----------------------------------------------------------------------------

Workflow
--------

:py:mod:`frites.workflow`:

.. currentmodule:: frites.workflow

.. automodule:: frites.workflow
   :no-members:
   :no-inherited-members:

Task-related workflows
++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   WfMi

Connectivity workflows
++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   WfConnComod


Statistical workflows
+++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   WfStats

.. raw:: html

  <hr>

.. ----------------------------------------------------------------------------

Connectivity
------------

:py:mod:`frites.conn`:

.. currentmodule:: frites.conn

.. automodule:: frites.conn
   :no-members:
   :no-inherited-members:

Connectivity metrics
++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   conn_dfc
   conn_covgc
   conn_transfer_entropy

Utility functions
+++++++++++++++++

.. autosummary::
   :toctree: generated/

   conn_reshape_undirected
   conn_reshape_directed
   conn_ravel_directed
   conn_get_pairs
   define_windows
   plot_windows

.. raw:: html

  <hr>

.. ----------------------------------------------------------------------------

Information-based estimators
----------------------------

:py:mod:`frites.estimator`:

.. currentmodule:: frites.estimator

.. automodule:: frites.estimator
   :no-members:
   :no-inherited-members:

Information-theoretic estimators
++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   GCMIEstimator
   BinMIEstimator

Distance estimators
+++++++++++++++++++

.. autosummary::
   :toctree: generated/

   DcorrEstimator

Correlation estimators
++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   CorrEstimator

.. raw:: html

  <hr>

.. ----------------------------------------------------------------------------

Simulations
-----------

:py:mod:`frites.simulations`:

.. currentmodule:: frites.simulations

.. automodule:: frites.simulations
   :no-members:
   :no-inherited-members:

Stimulus-specific autoregressive model
++++++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   StimSpecAR


Single and multi-subjects gaussian-based simulations
++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   sim_local_cc_ss
   sim_local_cc_ms
   sim_local_cd_ss
   sim_local_cd_ms
   sim_local_ccd_ms
   sim_local_ccd_ss

.. raw:: html

  <hr>

.. ----------------------------------------------------------------------------

.. note::

    This second section contains important internal functions for developers

Statistics
----------

:py:mod:`frites.stats`:

.. currentmodule:: frites.stats

.. automodule:: frites.stats
   :no-members:
   :no-inherited-members:

Random effect (rfx)
+++++++++++++++++++

.. autosummary::
   :toctree: generated/

   ttest_1samp
   rfx_ttest

Correction for multiple comparisons
+++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   cluster_correction_mcp
   testwise_correction_mcp
   cluster_threshold


.. raw:: html

  <hr>


.. ----------------------------------------------------------------------------


Utils
-----

:py:mod:`frites.utils`:

.. currentmodule:: frites.utils

.. automodule:: frites.utils
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   savgol_filter
   kernel_smoothing
   time_to_sample
   get_closest_sample

.. raw:: html

  <hr>

.. ----------------------------------------------------------------------------

Configuration
-------------

:py:mod:`frites.config`:

.. currentmodule:: frites

.. autosummary::
   :toctree: generated/

   get_config
   set_config

.. raw:: html

  <hr>

Core
----

:py:mod:`frites.core`:

.. currentmodule:: frites.core

.. automodule:: frites.core
   :no-members:
   :no-inherited-members:


Gaussian-Copula (1d)
++++++++++++++++++++

Gaussian-Copula mutual-information supporting univariate / multivariate 2D inputs

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

Gaussian-Copula mutual-information supporting univariate / multivariate multi-dimensional inputs

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
