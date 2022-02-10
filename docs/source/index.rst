Frites
======

.. image:: https://github.com/brainets/frites/actions/workflows/test_doc.yml/badge.svg
    :target: https://github.com/brainets/frites/actions/workflows/test_doc.yml

.. image:: https://github.com/brainets/frites/actions/workflows/flake.yml/badge.svg
    :target: https://github.com/brainets/frites/actions/workflows/flake.yml

.. image:: https://travis-ci.org/brainets/frites.svg?branch=master
    :target: https://travis-ci.org/brainets/frites

.. image:: https://codecov.io/gh/brainets/frites/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/brainets/frites

.. image:: https://badge.fury.io/py/frites.svg
    :target: https://badge.fury.io/py/frites

.. image:: https://pepy.tech/badge/frites
    :target: https://pepy.tech/project/frites



.. figure::  _static/logo_desc.png
    :align:  center

.. _MNE-Python: https://mne.tools/stable
.. _MNE-Connectivity: https://mne.tools/mne-connectivity/dev/

Description
+++++++++++

**Frites** is a Python toolbox for assessing information-based measures on human and animal neurophysiological data (M/EEG, Intracranial). The toolbox also includes directed and undirected connectivity metrics such as group-level statistics on measures of information (information-theory, machine-learning and measures of distance).
The toolbox builds off the popular `MNE-Python`_ and `MNE-Connectivity`_ packages to perform connectivity analyses.

Highlights
++++++++++

.. panels::
    :container: container-lg pb-1
    :header: text-center
    :card: shadow

    Measures of information
    ^^^^^^^^^^^^^^^^^^^^^^^

    Extract cognitive brain networks by linking brain data to an external task
    variable by means of powerful, potentially multivariate, measures of
    information from fields like `information theory <https://brainets.github.io/frites/api/api_estimators.html#information-theoretic-estimators>`_, machine-learning,
    `measures of distances <https://brainets.github.io/frites/api/api_estimators.html#distance-estimators>`_
    or by defining your own `custom estimator <https://brainets.github.io/frites/api/generated/frites.estimator.CustomEstimator.html#frites.estimator.CustomEstimator>`_.

    +++

    .. link-button:: https://brainets.github.io/frites/api/api_estimators.html
        :text: List of estimators
        :classes: btn-outline-primary btn-block

    ---

    Group-level statistics
    ^^^^^^^^^^^^^^^^^^^^^^

    Combine measures of information with group-level statistics to extract
    robust effects across a population of subjects or sessions. Use either fully
    automatized non-parametric permutation-based `workflows <https://brainets.github.io/frites/api/generated/frites.workflow.WfMi.html>`_ or `manual ones <https://brainets.github.io/frites/api/generated/frites.workflow.WfStats.html>`_

    +++

    .. link-button:: https://brainets.github.io/frites/api/api_workflow.html
        :text: List of workflows
        :classes: btn-outline-primary btn-block

    ---

    Measures of connectivity
    ^^^^^^^^^^^^^^^^^^^^^^^^

    Estimate whole-brain pairwise undirected and directed connectivity.

    +++

    .. link-button:: https://brainets.github.io/frites/api/api_connectivity.html
        :text: Connectivity metrics
        :classes: btn-outline-primary btn-block stretched-link

    ---

    Link with external toolboxes
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Frites supports inputs from standard libraries like `Numpy <https://numpy.org/>`_,
    `MNE Python <https://mne.tools/stable/index.html>`_ or more recent ones like
    labelled `Xarray <http://xarray.pydata.org/en/stable/>`_ objects.

    +++

    .. link-button:: https://brainets.github.io/frites/auto_examples/index.html#xarray
        :text: Xarray quick tour
        :classes: btn-outline-primary btn-block

.. toctree::
   :maxdepth: 2
   :hidden:

   Overview <overview/index>
   Installation <install>
   API Reference <api/index>
   Examples <auto_examples/index>
