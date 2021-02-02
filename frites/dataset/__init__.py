"""Multi-subjects container.

The datasets are used to merge the data coming from multiple subjects. Several
input types are supported (NumPy, MNE, Xarray).
"""
from .suj_ephy import SubjectEphy  # noqa
from .ds_ephy import DatasetEphy  # noqa
