API
===

.. contents::
   :local:
   :depth: 2


Workflow
--------

.. currentmodule:: frites.workflow

.. automodule:: frites.workflow
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   WorkflowMiStats

Dataset
-------

.. currentmodule:: frites.dataset

.. automodule:: frites.dataset
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   DatasetEphy


Simulations
-----------


.. currentmodule:: frites.simulations

.. automodule:: frites.simulations
   :no-members:
   :no-inherited-members:

Simulate random data
++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   sim_single_suj_ephy
   sim_multi_suj_ephy

Simulate random mutual information
++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   sim_mi_cc
   sim_mi_cd
   sim_mi_ccd

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