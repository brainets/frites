"""Containers of electrophysiological data.

This submodule includes containers for the neurophysiological data either for
a single-subject or multiple subjects. Several input types are supported
(NumPy, MNE, Neo, Xarray).
"""
from .suj_ephy import SubjectEphy  # noqa
from .ds_ephy import DatasetEphy  # noqa
