======
frites
======

.. image:: https://github.com/brainets/frites/workflows/frites/badge.svg
    :target: https://github.com/brainets/frites/workflows/frites

.. image:: https://travis-ci.org/brainets/frites.svg?branch=master
    :target: https://travis-ci.org/brainets/frites

.. image:: https://circleci.com/gh/brainets/frites.svg?style=svg
    :target: https://circleci.com/gh/brainets/frites

.. image:: https://codecov.io/gh/brainets/frites/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/brainets/frites

.. image:: https://badge.fury.io/py/frites.svg
    :target: https://badge.fury.io/py/frites

.. image:: https://pepy.tech/badge/frites
    :target: https://pepy.tech/project/frites

.. figure::  https://github.com/brainets/frites/blob/master/docs/source/_static/frites.png
    :align:  center


Description
-----------


**FRITES = Framework for Information Theoretical analysis of Electrophysiological data and Statistics**


Frites is a python package for analyzing neurophysiological brain data (i.e M/EEG, sEEG / iEEG / ECoG). The package is entirely based on information theoretic measures (such as mutual information (MI)) in order to perform analysis such as :

* "Correlation like" (**I(c; c)** = MI between two continuous variables)
* "Machine-learning like" (**I(c; d)** = MI between a continuous and a discrete variable)
* "Partial correlation like" (**I(c; c | d)** = MI between two continuous variables and removing the influence of a discrete one)
* Information-transfer about a specific feature

For a comprehensive (and extensive) review, see the paper of Robin AA Ince `A statistical framework for neuroimaging data analysis based on mutual information estimated via a gaussian copula <https://www.ncbi.nlm.nih.gov/pubmed/27860095>`_.

Frites also comes with embedded statistics which support fixed and random-effect analysis in combination with inferences either at the single time-point level or at the temporal cluster level.

Take a look at the online documentation and examples to start analyzing your data with Frites : https://brainets.github.io/frites/


Installation
------------

Dependencies
++++++++++++

The main dependencies of Frites are :

* `Numpy <https://numpy.org/>`_
* `Scipy <https://www.scipy.org/>`_
* `MNE <https://mne.tools/stable/index.html>`_
* `Joblib <https://joblib.readthedocs.io/en/latest/>`_

In addition to the main dependencies, here's the list of additional packages that you might need :

* `Pandas <https://pandas.pydata.org/>`_ and `Xarray <http://xarray.pydata.org/en/stable/>`_ : additional output types
* `Numba <http://numba.pydata.org/>`_ : speed computations of some functions


User installation
+++++++++++++++++

Frites can be installed (and/or updated) via pip with the following command :

.. code-block:: shell

    pip install -U frites


Developer installation
++++++++++++++++++++++

For developers, you can install frites in develop mode with the following commands :

.. code-block:: shell

    git clone https://github.com/brainets/frites.git
    cd frites
    python setup.py develop

