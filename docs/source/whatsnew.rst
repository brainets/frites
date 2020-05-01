.. currentmodule:: frites

What's new
==========

.. raw:: html

    <hr>

v0.3.1
------

Breaking changes
++++++++++++++++
* change :py:class:`frites.workflow.WfFit` input `directed` for `net` (:commit:`3ec0`, :issue:`1`)
* The GCRN is automatically defined (per subject when RFX / across subjects when FFX) (:commit:`b8a9`)

New Features
++++++++++++
* :py:class:`frites.dataset.DatasetEphy` support spatio-temporal slicing (:commit:`60d1`), resampling (:commit:`3785`) and savitzki-golay filter (:commit:`9707`)
* Support setting `random_state` in :py:class:`frites.workflow.WfMi` and :py:class:`frites.workflow.WfFit` (:commit:`0688`)
* DataArray outputs contains attributes that reflect the configuration of the workflow (:commit:`18181`)

Bug fixes
+++++++++
* Fix multi-sites concatenation (:commit:`3bcc`) in :py:class:`frites.dataset.DatasetEphy`
* Fix p-values to zeros in :py:class:`frites.workflow.WfFit` (:commit:`2062`)
* Fix FIT outputs for 3D arrays and DataArray (:commit:`18181`)

Internal Changes
++++++++++++++++
* Remap multiple conditions when integers (:commit:`6092`) in :py:class:`frites.dataset.DatasetEphy`
* Workflows now have an internal configuration (:commit:`18181`)

Documentation
+++++++++++++
* Reformat examples gallery (:commit:`4d4c`)
