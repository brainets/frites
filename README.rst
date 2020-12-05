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
* `Xarray <http://xarray.pydata.org/en/stable/>`_
* `Joblib <https://joblib.readthedocs.io/en/latest/>`_

In addition to the main dependencies, here's the list of additional packages that you might need :

* `Numba <http://numba.pydata.org/>`_ : speed computations of some functions
* `Matplotlib <https://matplotlib.org/>`_, `Seaborn <https://seaborn.pydata.org/>`_ and `Networkx <https://networkx.github.io/>`_ for plotting the examples


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


Contributors ✨
--------------

Thanks goes to these wonderful people (`emoji key`_):

.. _emoji key: https://allcontributors.org/docs/en/emoji-key

.. raw:: html

   <!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
  <!-- prettier-ignore-start -->
  <!-- markdownlint-disable -->
  <table>
    <tr>
      <td align="center"><a href="https://github.com/EtienneCmb"><img src="https://avatars3.githubusercontent.com/u/15892073?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Etienne Combrisson</b></sub></a><br /><a href="https://github.com/brainets/frites/commits?author=EtienneCmb" title="Code">💻</a> <a href="#design-EtienneCmb" title="Design">🎨</a> <a href="#example-EtienneCmb" title="Examples">💡</a> <a href="#maintenance-EtienneCmb" title="Maintenance">🚧</a> <a href="#mentoring-EtienneCmb" title="Mentoring">🧑‍🏫</a> <a href="#projectManagement-EtienneCmb" title="Project Management">📆</a></td>
      <td align="center"><a href="http://andrea-brovelli.net/"><img src="https://avatars0.githubusercontent.com/u/19585963?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Andrea Brovelli</b></sub></a><br /><a href="https://github.com/brainets/frites/commits?author=brovelli" title="Code">💻</a> <a href="#ideas-brovelli" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-brovelli" title="Mentoring">🧑‍🏫</a> <a href="#projectManagement-brovelli" title="Project Management">📆</a></td>
      <td align="center"><a href="https://github.com/StanSStanman"><img src="https://avatars1.githubusercontent.com/u/26648765?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ruggero Basanisi</b></sub></a><br /><a href="https://github.com/brainets/frites/commits?author=StanSStanman" title="Code">💻</a> <a href="#design-StanSStanman" title="Design">🎨</a></td>
      <td align="center"><a href="https://github.com/ViniciusLima94"><img src="https://avatars3.githubusercontent.com/u/17538901?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Vinícius Lima</b></sub></a><br /><a href="https://github.com/brainets/frites/commits?author=ViniciusLima94" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/tprzybyl"><img src="https://avatars1.githubusercontent.com/u/58084045?v=4?s=100" width="100px;" alt=""/><br /><sub><b>tprzybyl</b></sub></a><br /><a href="https://github.com/brainets/frites/commits?author=tprzybyl" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/micheleallegra"><img src="https://avatars0.githubusercontent.com/u/23451833?v=4?s=100" width="100px;" alt=""/><br /><sub><b>micheleallegra</b></sub></a><br /><a href="https://github.com/brainets/frites/commits?author=micheleallegra" title="Code">💻</a> <a href="#ideas-micheleallegra" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="http://www.robinince.net/"><img src="https://avatars0.githubusercontent.com/u/63155?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Robin Ince</b></sub></a><br /><a href="https://github.com/brainets/frites/commits?author=robince" title="Code">💻</a> <a href="#ideas-robince" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/samuelgarcia"><img src="https://avatars1.githubusercontent.com/u/815627?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Garcia Samuel</b></sub></a><br /><a href="#ideas-samuelgarcia" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </table>

  <!-- markdownlint-restore -->
  <!-- prettier-ignore-end -->

  <!-- ALL-CONTRIBUTORS-LIST:END -->