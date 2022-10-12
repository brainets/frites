"""Base class for managing Xarray attributes."""
from collections import UserDict

import numpy as np
import xarray as xr
from xarray.core import formatting, formatting_html


class Attributes(UserDict):

    """Base class attributes."""

    def __init__(self, attrs=None, section_name='Attributes'):
        """Init."""
        if not isinstance(attrs, dict):
            attrs = dict()
        self.section_name = section_name
        UserDict.__init__(self, attrs)

    def __getitem__(self, key):
        """Get attribute items."""
        return self.data[key]

    def __setitem__(self, key, value):
        """Set attribute items."""
        self.data[key] = 'none' if value is None else value

    def __repr__(self):
        """Global representation."""
        return formatting.attrs_repr(self.data)

    def _repr_html_(self):
        """HTML representation."""
        data = {str(k): i for k, i in self.data.items()}
        sections = [formatting_html.attr_section(data)]
        return formatting_html._obj_repr(None, self.section_name, sections)

    def _check_netcdf(self):
        """Check attributes for netcdf compatibility."""
        self.data = check_attrs(self.data)

    def update(self, attrs, check=True):
        """Update internal with external attributes."""
        self.data.update(attrs)
        if check:
            self._check_netcdf()

    def merge(self, list_attrs):
        """Merge a list of multiple attributes."""
        assert isinstance(list_attrs, list)
        if not len(list_attrs):
            return
        for k in list_attrs:
            assert isinstance(k, dict)
            self.update(k, check=False)
        self._check_netcdf()

    def wrap_xr(self, da, name=None, **kwargs):
        """Wrap a data array / set with internal attributes."""
        assert isinstance(da, (xr.DataArray, xr.Dataset))
        self.merge([da.attrs, kwargs])
        da.attrs = self.data
        if isinstance(name, str):
            da.name = name
            da.attrs['type'] = name
        return da


def check_attrs(kwargs):
    """Check Xarray attributes for loading / saving files.

    Saving Xarray as netcdf can sometimes fail due to bad attribute types. This
    function convert those problematic attributes.

    Parameters
    ----------
    kwargs : dict
        Attributes to convert

    Returns
    -------
    attrs : dict
        Converted attributes
    """
    assert isinstance(kwargs, dict)
    attrs = {}
    for k, v in kwargs.items():
        # None to 'none'
        if v is None:
            v = 'None'

        # bool to int
        if isinstance(v, bool):
            v = int(v)

        # array to list
        if isinstance(v, np.ndarray):
            attrs[k] = np.ravel(v).tolist()

        # dict to strings
        v_c = {}
        if isinstance(v, dict):
            for _k, _v in v.items():
                v_c[f"{k}_{_k}"] = _v
            v = v_c

        # finally set the converted attribute
        attrs[k] = v

    return attrs
