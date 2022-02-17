.. currentmodule:: frites

What's new
==========

v0.4.2
------

New Features
++++++++++++
* New function :func:`frites.simulations.sim_ground_truth` for simulating spatio-temporal ground-truths (:commit:`ba44a424`)
* New function :func:`frites.conn.conn_spec` for computing the single-trial spectral connectivity (:commit:`8151486`) - :author:`ViniciusLima94`
* New method :class:`frites.workflow.WfMi.confidence_interval` method to estimate the confidence interval (:commit:`ad0391987`, :commit:`8189622b`, :commit:`fc584756`, :commit:`fc584756`)
* New function :func:`frites.conn.conn_net` for computing the net connectivity (:commit:`c86b19f0`)
* New function :func:`frites.set_mpl_style` for example styles
* New function :func:`frites.conn.conn_links` for generating connectivity links (:commit:`a0d0182d1`)

v0.4.1
------

New Features
++++++++++++
* New :class:`frites.estimator.CustomEstimator` for defining custom estimators (:commit:`e473c713`, :commit:`5584654c`)
* New function :func:`frites.conn.conn_fcd_corr` for computing the temporal correlation across networks (:commit:`2001f0c0`)
* New function :func:`frites.utils.acf` for computing the auto-correlation (:commit:`48ef0a03`)
* New function :func:`frites.conn.conn_ccf` for computing the cross-correlation (:commit:`43fceb00`)

Bug fixes
+++++++++
* Fix attribute conversion in connectivity functions (:commit:`b990c76`)

v0.4.0
------

New Features
++++++++++++
* New estimators (:class:`frites.estimator.CorrEstimator`, :class:`frites.estimator.DcorrEstimator`) for continuous / continuous relationships (:commit:`73ed8bbb`, :commit:`bc370a93`, :commit:`cf7a3456f`)
* :func:`frites.conn.conn_dfc` supports passing other estimators (:commit:`a864a7b05b`)
* :func:`frites.utils.time_to_sample` and :func:`frites.utils.get_closest_sample` conversion functions (:commit:`7c44478e`)
* :func:`frites.conn.conn_ravel_directed` reshaping function (:commit:`f9b9d272`)
* New :class:`frites.workflow.WfMi.copy` for internal workflow copy (:commit:`0c2228c7`, :commit:`860f3d45`)
* New :class:`frites.workflow.WfMiCombine` and example class for combining workflows (:commit:`62072ee52`)
* New :class:`frites.estimator.ResamplingEstimator` trial-resampling estimator (:commit:`13f6271e`)

Bug fixes
+++++++++
* Fix :class:`frites.workflow.WfMi.get_params` when returning a single output (:commit:`3bde82e6`)
* Improve attributes conversion for saving netcdf files (bool and dict) (:commit:`8e7dddb1`, :commit:`c6f7a4db`)
* Fix Numpy `np.float` and `np.int` warnings related (:commit:`896a198a`, :commit:`7f2a1caef`, :commit:`0d1a1223`)

v0.3.9
------

New Features
++++++++++++
* :func:`frites.conn.conn_dfc` supports multivariate data + improve computing efficiency (:commit:`1aed842`, :commit:`c4ac490`)
* Reshaping connectivity arrays support elements on the diagonal + internal drop of duplicated elements (:commit:`daac241f`)
* :func:`frites.conn.conn_dfc` supports better channel aggregation (:commit:`a66faa77`)

Internal Changes
++++++++++++++++
* Connectivity metric now use the :class:`frites.dataset.SubjectEphy` for internal conversion of the input data
* :class:`frites.workflow.WfMi.get_params` returns single-subject MI and permutations with dimension name 'subject' (instead of subjects) (:commit:`85884f3a`)
* All connectivity metrics now use :func:`frites.conn.conn_io` to convert inputs into a similar format
* Improve CI

Bug fixes
+++++++++
* Fix :class:`frites.dataset.SubjectEphy` when the data contains a single time point (:commit:`a33d4437`)
* Fix attributes of :func:`frites.conn.conn_covgc` (:commit:`c120626`)
* Fix :class:`frites.dataset.DatasetEphy` representation without data copy + html representation (:commit:`b3ae7b8ea`, :issue:`16`)
* Fix passing `tail` input to the :class:`frites.workflow.WfMi` (:commit:`6df86d1e`)


v0.3.8
------

New Features
++++++++++++
* new :class:`frites.io.Attributes` class for managing and printing datasets' and workflow's attributes (:commit:`be046b1`)
* new :class:`frites.dataset.SubjectEphy` single-subject container (:commit:`ac22cf4`)
* new estimators of mutual-information, :class:`frites.estimator.GCMIEstimator` (:commit:`901b3cbf`, :commit:`65d1e08`, :commit:`0015bf58`, :commit:`beed6a09`), :class:`frites.estimator.BinMIEstimator` (:commit:`beed6a09`)
* new kernel smoothing function :func:`frites.utils.kernel_smoothing`

Internal Changes
++++++++++++++++
* Removed files (:commit:`cdff9b4`, :commit:`9e96f8e`, :commit:`14961aa0`)
* :class:`frites.dataset.DatasetEphy` don't perform internal data copy when getting the data in a specific ROI (:commit:`2da73ef`)
* Compatibility of MI estimators with workflows (:commit:`7dc76ee9`, :commit:`e7a9c23f`)
* Improve the way to manage pairs of brain regions (:commit:`8b955a16`, :commit:`bfdf2dba`, :commit:`57c1e4ba`, :commit:`b1ff8c3d`)

Breaking changes
++++++++++++++++
* :class:`frites.dataset.SubjectEphy` and :class:`frites.dataset.DatasetEphy` to specify whether channels should be aggregated (default `agg_ch=True`) or not (`agg=False`) when computing MI. The `agg_ch` replace `sub_roi` (:commit:`18d4e24`)
* The workflow `WfComod` has been renamed :class:`frites.workflow.WfConnComod` (:commit:`b7b58248`)

Bug fixes
+++++++++
* Bug fixing according to the new version of :class:`frites.dataset.DatasetEphy` (:commit:`1a15e05`, :commit:`7b83a3d`, :commit:`abd1b281`, :commit:`70bfefb`, :commit:`5879950`, :commit:`66acdf2`, :commit:`4309be9c5`, :commit:`6dc2fbf8`)

v0.3.6
------

New Features
++++++++++++
* :class:`frites.dataset.DatasetEphy` support multi-level anatomical informations (:commit:`3a9ce540`)

Internal Changes
++++++++++++++++
* Connectivity functions have a better support of Xarray inputs (:commit:`60cc16`, :commit:`b72d1519`, :commit:`3d24c98`, :commit:`65bf08`)
* Replace every string comparison 'is' with '==' to test the content (:commit:`1337aa6e`)

v0.3.5
------

New Features
++++++++++++
* New function for reshaping undirected connectivity arrays (like DFC) :func:`frites.conn.conn_reshape_undirected` (:commit:`ffcae34`, :commit:`56515fe`)
* New function :func:`frites.utils.savgol_filter` that works on DataArray (:commit:`3e0e256`)
* New function for reshaping directed connectivity arrays (like COVGC) :func:`frites.conn.conn_reshape_directed` (:commit:`8c2bb63`)
* New method :class:`frites.workflow.WfMi.get_params` in order to get the internal arrays formatted as DataArray (:commit:`03dd2f3`)
* Integration of MNE's progress bar (:commit:`74dc66`, :commit:`8ec636d`, :commit:`2bb7e75`)
* Possibility to cache computations in the parallel function

Internal Changes
++++++++++++++++
* DataArray now contain a name such as a type to make it clear what is it (:commit:`2d6a61`)
* sigma parameter when performing the t-test can be changed though the CONFIG file (:commit:`5d7ba9f`)

Bug fixes
+++++++++
* Fix ttested attribute when saving (:commit:`3d029be`)
* Fix computing the sigma across all ROI since it uses the maximum over every axes (:commit:`ffaca1e`)
* Fix high RAM consumption when computing the `pop_mean_surr` (:commit:`fe6e4d1`)


.. raw:: html

    <hr>

v0.3.4
------

Bug fixes
+++++++++
* Fix :class:`frites.workflow.WfMi.conjunction_analysis` for seeg data (:commit:`6ca64c5`)

Breaking changes
++++++++++++++++
* Xarray is now the default output type as it supports the definition of multi-dimensional containers with label to each coordinates (:commit:`8174e4`) + illustrating examples of how to use xarray (:commit:`f5d28e`)
* Every connectivity measures have been moved to `frites.conn` (:commit:`a33d536`)
* Deep support for correcting for multiple-comparisons using multi-dimensional data (:commit:`840327`, :commit:`e0e1a5b`) + support for :class:`frites.workflow.WfStatsEphy` (:commit:`563910d`) + support for :class:`frites.dataset.DatasetEphy` (:commit:`10a3697`) + support for :class:`frites.workflow.WfMi` (:commit:`9c6165`)

New Features
++++++++++++
* New class :class:`frites.simulations.StimSpecAR` for generating Auto-Regressive Models (:commit:`4688539`, :commit:`5fd1199`)
* Conditional Covariance based Granger Causality + example (:commit:`a148310`)

Internal Changes
++++++++++++++++
* Improve testings (:commit:`6644cd`)

.. raw:: html

    <hr>

v0.3.3
------

Internal Changes
++++++++++++++++
* :class:`frites.workflow.WfFit` and :class:`frites.workflow.WfConn` are now using :class:`frites.dataset.DatasetEphy.get_connectivity_pairs` (:commit:`18e1`)
* Improve warning messages + assert error for negative determinant for :class:`frites.core.covgc`

New Features
++++++++++++
* New method :class:`frites.dataset.DatasetEphy.get_connectivity_pairs` in order to get possible connectivity pairs (:commit:`4e77`)
* New function :func:`frites.utils.define_windows` and :func:`frites.utils.plot_windows` in order to generate and plot slicing windows + tests + example (:commit:`6fcf`)
* New function for computing the DFC :class:`frites.core.dfc_gc` (:commit:`8f6f`)
* When using DataArray with :class:`frites.core.dfc_gc` and :class:`frites.core.covgc`, temporal attributes are added (:commit:`5df6`, :commit:`6266`)
* New function for computing the covgc :class:`frites.core.covgc` (:commit:`ea26`)
* Step parameter for :class:`frites.core.covgc` (:commit:`874af`)
* :class:`frites.core.covgc` can no be computed using Gaussian-Copula (:commit:`aea6a8b`)
* Add :class:`frites.workflow.WfMi.conjunction_analysis` for performing conjunction analysis + example (:commit:`7dd0bea`)

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
* :class:`frites.dataset.DatasetEphy` supports None for the y input (:commit:`a53a`)

Internal Changes
++++++++++++++++
* Dtypes of `y` and `z` inputs are systematically check in :class:`frites.dataset.DatasetEphy` in order to define which MI can then be computed (:commit:`7cc1`)

New Features
++++++++++++
* :class:`frites.dataset.DatasetEphy` supports Xarray inputs + selection though coordinates (:commit:`7418`)
* New workflow for computing pairwise connectivity :class:`frites.workflow.WfConn` (:commit:`65ae`)

Documentation
+++++++++++++
* Adding new examples for creating datasets (:commit:`4df9`)


.. raw:: html

    <hr>

v0.3.1
------

Breaking changes
++++++++++++++++
* change :class:`frites.workflow.WfFit` input `directed` for `net` (:commit:`3ec0`, :issue:`1`)
* The GCRN is automatically defined (per subject when RFX / across subjects when FFX) (:commit:`b8a9`)
* Remove the `level` input parameter + only `mcp` is used + only maxstat when performing cluster based (:commit:`d8fb`)

New Features
++++++++++++
* :class:`frites.dataset.DatasetEphy` support spatio-temporal slicing (:commit:`60d1`), resampling (:commit:`3785`) and savitzki-golay filter (:commit:`9707`)
* Support setting `random_state` in :class:`frites.workflow.WfMi` and :class:`frites.workflow.WfFit` (:commit:`0688`)
* DataArray outputs contains attributes that reflect the configuration of the workflow (:commit:`18181`)

Bug fixes
+++++++++
* Fix multi-sites concatenation (:commit:`3bcc`) in :class:`frites.dataset.DatasetEphy`
* Fix p-values to zeros in :class:`frites.workflow.WfFit` (:commit:`2062`)
* Fix FIT outputs for 3D arrays and DataArray (:commit:`18181`)

Internal Changes
++++++++++++++++
* Remap multiple conditions when integers (:commit:`6092`) in :class:`frites.dataset.DatasetEphy`
* Workflows now have an internal configuration (:commit:`18181`)

Documentation
+++++++++++++
* Reformat examples gallery (:commit:`4d4c`)
