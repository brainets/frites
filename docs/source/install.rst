Installation
------------

.. contents::
   :local:
   :depth: 2

Dependencies
++++++++++++

The main dependencies of Frites are :

* `Numpy <https://numpy.org/>`_
* `Scipy <https://www.scipy.org/>`_
* `MNE <https://mne.tools/stable/index.html>`_
* `Neo <https://pypi.org/project/neo>`_
* `Xarray <http://xarray.pydata.org/en/stable/>`_
* `Joblib <https://joblib.readthedocs.io/en/latest/>`_

In addition to the main dependencies, here's the list of additional packages that you might need :

* `Numba <http://numba.pydata.org/>`_ : speed computations of some functions
* `Matplotlib <https://matplotlib.org/>`_, `Seaborn <https://seaborn.pydata.org/>`_ and `Networkx <https://networkx.github.io/>`_ for plotting the examples


Installation from pip
+++++++++++++++++++++

Frites can be installed (and/or updated) via pip with the following command :

.. code-block:: shell

    pip install -U frites

If you choose this installation method, you'll get stable Frites' release. If you want the latest features and patches, you can directly install from Github (see bellow)


Installation from Github
++++++++++++++++++++++++

Run the following line in your terminal to install the latest version of Frites hosted on Github :

.. code-block:: shell

    pip install git+https://github.com/brainets/frites.git


Developer installation
++++++++++++++++++++++

For developers, you can install frites in develop mode with the following commands :

.. code-block:: shell

    git clone https://github.com/brainets/frites.git
    cd frites
    python setup.py develop
    # or : pip install -e .