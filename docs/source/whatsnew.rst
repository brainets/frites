.. currentmodule:: frites

What's new
==========

v0.3.4
------

Breaking changes
++++++++++++++++
* Xarray is now the default output type as it supports the definition of multi-dimensional containers with label to each coordinates (:commit:`8174e4`) + illustrating examples of how to use xarray (:commit:`f5d28e`)
* Every connectivity measures have been moved to `frites.conn` (:commit:`a33d536`)
* Deep support for correcting for multiple-comparisons using multi-dimensional data (:commit:`840327`, :commit:`e0e1a5b`)

v0.3.3
------

Internal Changes
++++++++++++++++
* :py:class:`frites.workflow.WfFit` and :py:class:`frites.workflow.WfConn` are now using :py:class:`frites.dataset.DatasetEphy.get_connectivity_pairs` (:commit:`18e1`)
* Improve warning messages + assert error for negative determinant for :py:class:`frites.core.covgc`

New Features
++++++++++++
* New method :py:class:`frites.dataset.DatasetEphy.get_connectivity_pairs` in order to get possible connectivity pairs (:commit:`4e77`)
* New function :py:func:`frites.utils.define_windows` and :py:func:`frites.utils.plot_windows` in order to generate and plot slicing windows + tests + example (:commit:`6fcf`)
* New function for computing the DFC :py:class:`frites.core.dfc_gc` (:commit:`8f6f`)
* When using DataArray with :py:class:`frites.core.dfc_gc` and :py:class:`frites.core.covgc`, temporal attributes are added (:commit:`5df6`, :commit:`6266`)
* New function for computing the covgc :py:class:`frites.core.covgc` (:commit:`ea26`)
* Step parameter for :py:class:`frites.core.covgc` (:commit:`874af`)
* :py:class:`frites.core.covgc` can no be computed using Gaussian-Copula (:commit:`aea6a8b`)
* Add :py:class:`frites.workflow.WfMi.conjunction_analysis` for performing conjunction analysis + example (:commit:`7dd0bea`)

Bug fixes
+++++++++
* Fix when data contains a single time point (:commit:`c815e40`)
* Fix mi model and mixture (1d only) (:commit:`2b119d4`)

.. raw:: html

    <hr>

v0.3.2
------

Breaking changes
++++++++++++++++
* Avoid duplicates dataset construction when using MNE / xarray (:commit:`fef9`, :commit:`9dde`)
* :py:class:`frites.dataset.DatasetEphy` supports None for the y input (:commit:`a53a`)

Internal Changes
++++++++++++++++
* Dtypes of `y` and `z` inputs are systematically check in :py:class:`frites.dataset.DatasetEphy` in order to define which MI can then be computed (:commit:`7cc1`)

New Features
++++++++++++
* :py:class:`frites.dataset.DatasetEphy` supports Xarray inputs + selection though coordinates (:commit:`7418`)
* New workflow for computing pairwise connectivity :py:class:`frites.workflow.WfConn` (:commit:`65ae`)

Documentation
+++++++++++++
* Adding new examples for creating datasets (:commit:`4df9`)


.. raw:: html

    <hr>

v0.3.1
------

Breaking changes
++++++++++++++++
* change :py:class:`frites.workflow.WfFit` input `directed` for `net` (:commit:`3ec0`, :issue:`1`)
* The GCRN is automatically defined (per subject when RFX / across subjects when FFX) (:commit:`b8a9`)
* Remove the `level` input parameter + only `mcp` is used + only maxstat when performing cluster based (:commit:`d8fb`)

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
