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


Description
+++++++++++

**Frites** is a Python toolbox for assessing information-based measures on human and animal neurophysiological data (M/EEG, Intracranial). The toolbox also includes directed and undirected connectivity metrics such as group-level statistics on measures of information (information-theory, machine-learning and measures of distance).

What can you do with Frites?
++++++++++++++++++++++++++++

1. Frites can extract task-related cognitive brain networks, that is brain regions and connectivity between brain regions that are modulated according to the task (`overview <https://brainets.github.io/frites/overview/ovw_goals.html#>`_)
2. Frites can assess information-based measures on neurophysiological data (`overview <https://brainets.github.io/frites/overview/ovw_methods.html#data-analysis-within-the-information-theoretical-framework>`_) using a large panel of estimators (`examples <https://brainets.github.io/frites/auto_examples/index.html#group-level-statistics-on-measures-of-information>`_)
3. Frites can be used to assess Dynamic Functional Connectivity using mutual information and directional measures using Granger causality (`examples <https://brainets.github.io/frites/auto_examples/index.html#connectivity-and-information-transfer>`_)
4. Frites can perform statistical inference on measures of information, such as measures from information-theory, machine-learning or measures of distances, using permutation-based tests (`overview <https://brainets.github.io/frites/overview/ovw_methods.html#statistical-analyses>`_ and `examples <https://brainets.github.io/frites/auto_examples/index.html#group-level-statistics-on-measures-of-information>`_) and controlling for multiple comparisons (`overview <https://brainets.github.io/frites/overview/ovw_methods.html#correction-for-multiple-comparisons>`_)
5. Frites provides simples WorkFlows to analyse and organise your datasets (`overview <https://brainets.github.io/frites/overview/ovw_frites.html#start-analyzing-your-data-with-frites>`_ and `examples <https://brainets.github.io/frites/auto_examples/index.html#multi-subjects-dataset>`_) and simple tutorials (`examples <https://brainets.github.io/frites/auto_examples/index.html#tutorials>`_)


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
