API
===

This section contains the list modules, classes and functions of Frites.

Overview
++++++++

.. panels::
    :container: container-lg pb-1
    :header: text-center
    :card: shadow

    Dataset
    ^^^^^^^

    The :mod:`frites.dataset` module contains data containers (a similar idea
    to MNE-Python with objects like :mod:`mne.Raw` or :mod:`mne.Epochs`) for
    single and multi subjects

    ---

    Workflows
    ^^^^^^^^^

    The :mod:`frites.workflow` module contains automated pipelines that are
    usually composed of two steps : `(i)` a first step consisting in estimating
    a measure of effect size (e.g. modulation of neural activity according to
    task conditions) followed by `(ii)` a statistical layer

    ---
    :column: col-lg-12 p-2

    Connectivity
    ^^^^^^^^^^^^

    The :mod:`frites.conn` module contains functions to estimate undirected and
    directed functional connectivity, potentially dynamic, at the single trial
    level

    ---
    :column: col-lg-12 p-2

    Estimators
    ^^^^^^^^^^

    The :mod:`frites.estimator` module contains information-based estimators
    for linking brain data to an external variable (e.g. stimulus type,
    behavioral models etc.). This includes metrics from the information-theory,
    machine-learning or measures of distances

    ---
    :column: col-lg-12 p-2

    Plot
    ^^^^

    The :mod:`frites.plot` module contains plotting functions
    ---

    Simulated data
    ^^^^^^^^^^^^^^

    The :mod:`frites.simulations` module contains functions and classes to
    generate simulated data (single or multi-subjects)

    ---

    Utils
    ^^^^^

    The :mod:`frites.utils` module contains utility functions (e.g. data
    smoothing)


List of classes and functions
+++++++++++++++++++++++++++++

High-level API for users
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    api_dataset
    api_workflow
    api_connectivity
    api_estimators
    api_plot
    api_simulations
    api_utils

Low-level API for developers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    api_statistics
    api_core
    api_conf
