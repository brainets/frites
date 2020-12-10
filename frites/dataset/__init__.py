"""Multi-subjects container.

The datasets are used to merge the data coming from multiple subjects. Several
input types are supported (NumPy, MNE, Xarray).
"""
from .ds_ephy import DatasetEphy  # noqa
from .ds_fmri import DatasetFMRI  # noqa
