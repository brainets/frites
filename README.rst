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

.. figure::  https://github.com/brainets/frites/blob/master/docs/source/_static/logo_desc.png
    :align:  center

======
Frites
======

Description
-----------

`Frites <https://brainets.github.io/frites/>`_ is a Python toolbox for assessing information-theorical measures on human and animal neurophysiological data (M/EEG, Intracranial). The aim of Frites is to extract task-related cognitive brain networks (i.e modulated by the task). The toolbox also includes directed and undirected connectivity metrics such as group-level statistics.

.. figure::  https://github.com/brainets/frites/blob/master/docs/source/_static/network_framework.png
    :align:  center

Documentation
-------------

Frites documentation is available online at https://brainets.github.io/frites/

Installation
------------

Run the following command into your terminal to get the latest stable version :

.. code-block:: shell

    pip install -U frites


You can also install the latest version of the software directly from Github :

.. code-block:: shell

    pip install git+https://github.com/brainets/frites.git


For developers, you can install it in develop mode with the following commands :

.. code-block:: shell

    git clone https://github.com/brainets/frites.git
    cd frites
    python setup.py develop
    # or : pip install -e .

Dependencies
++++++++++++

The main dependencies of Frites are :

* `Numpy <https://numpy.org/>`_
* `Scipy <https://www.scipy.org/>`_
* `MNE Python <https://mne.tools/stable/index.html>`_
* `Xarray <http://xarray.pydata.org/en/stable/>`_
* `Joblib <https://joblib.readthedocs.io/en/latest/>`_

In addition to the main dependencies, here's the list of additional packages that you might need :

* `Numba <http://numba.pydata.org/>`_ : speed up the computations of some functions
* `Dcor <https://dcor.readthedocs.io/en/latest/>`_ for fast implementation of distance correlation
* `Matplotlib <https://matplotlib.org/>`_, `Seaborn <https://seaborn.pydata.org/>`_ and `Networkx <https://networkx.github.io/>`_ for plotting the examples
* Some example are using `scikit learn <https://scikit-learn.org/stable/index.html>`_ estimators

Acknowledgments
---------------

See `acknowledgments <https://brainets.github.io/frites/overview/ovw_acknowledgments.html>`_
