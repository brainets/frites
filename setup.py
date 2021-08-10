#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: 3-clause BSD
import os
from setuptools import setup, find_packages

__version__ = "0.4.0"
NAME = 'frites'
AUTHOR = "BraiNets"
MAINTAINER = "Etienne Combrisson"
EMAIL = 'e.combrisson@gmail.com'
KEYWORDS = "information-theory statistics"
DESCRIPTION = ("Framework of Information Theory for Electrophysiological data "
               "and Statistics")
URL = 'https://github.com/brainets/frites'
DOWNLOAD_URL = ("https://github.com/brainets/frites/archive/v" +
                __version__ + ".tar.gz")
# Data path :
PACKAGE_DATA = {}


def read(fname):
    """Read README and LICENSE."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

core_deps = ['matplotlib', 'networkx', 'numba', 'dcor']
test_deps = ['pytest', 'pytest-sugar', 'pytest-cov', 'codecov']
doc_deps = [
    'sphinx==2.2.0', 'sphinx-gallery==0.4.0', 'sphinx_bootstrap_theme',
    'sphinxcontrib-bibtex==1.0.0', 'numpydoc', 'xlrd', 'openpyxl', 'seaborn'
]
flake_deps = ['flake8', 'pep8-naming']

setup(
    name=NAME,
    version=__version__,
    packages=find_packages(),
    package_dir={'frites': 'frites'},
    package_data=PACKAGE_DATA,
    include_package_data=True,
    description=DESCRIPTION,
    long_description=read('README.rst'),
    platforms='any',
    setup_requires=['numpy'],
    install_requires=requirements,
    extras_require={
        'all': core_deps,
        'test': core_deps + test_deps,
        'doc': core_deps + test_deps + doc_deps,
        'flake': core_deps + test_deps + flake_deps,
        'full': core_deps + test_deps + doc_deps + flake_deps
    },
    dependency_links=[],
    author=AUTHOR,
    maintainer=MAINTAINER,
    author_email=EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    license="BSD 3-Clause License",
    keywords=KEYWORDS,
    classifiers=["Development Status :: 5 - Production/Stable",
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'Intended Audience :: Developers',
                 "Programming Language :: Python :: 3.7",
                 "Programming Language :: Python :: 3.8"
                 ])
