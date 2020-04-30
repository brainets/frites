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

New Features
++++++++++++
* :py:class:`frites.dataset.DatasetEphy` support spatio-temporal slicing (:commit:`60d1`), resampling (:commit:`3785`) and savitzki-golay filter (:commit:`9707`)
* Support setting `random_state` in :py:class:`frites.workflow.WfMi` and :py:class:`frites.workflow.WfFit` (:commit:`0688`)

Bug fixes
+++++++++
* Fix multi-sites concatenation (:commit:`3bcc`) in :py:class:`frites.dataset.DatasetEphy`
* Fix p-values to zeros in :py:class:`frites.workflow.WfFit` (:commit:`2062`)

Internal Changes
++++++++++++++++
* Remap multiple conditions when integers (:commit:`6092`) in :py:class:`frites.dataset.DatasetEphy`

Documentation
+++++++++++++
* Reformat examples gallery (:commit:`4d4c`)
