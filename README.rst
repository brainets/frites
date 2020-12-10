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
* `Matplotlib <https://matplotlib.org/>`_, `Seaborn <https://seaborn.pydata.org/>`_ and `Networkx <https://networkx.github.io/>`_ for plotting the examples


Contributors ✨
---------------

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
      <td align="center"><a href="https://github.com/brungio"><img src="https://avatars0.githubusercontent.com/u/33055790?v=4?s=100" width="100px;" alt=""/><br /><sub><b>brungio</b></sub></a><br /><a href="https://github.com/brainets/frites/commits?author=brungio" title="Code">💻</a> <a href="#ideas-brungio" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-brungio" title="Mentoring">🧑‍🏫</a> <a href="#projectManagement-brungio" title="Project Management">📆</a></td>
    </tr>
  </table>

  <!-- markdownlint-restore -->
  <!-- prettier-ignore-end -->

  <!-- ALL-CONTRIBUTORS-LIST:END -->