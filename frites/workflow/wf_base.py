"""Base class for worflows."""
from frites.io import Attributes


class WfBase(object):
    """Base class for workflows."""

    def __init__(self, attrs=None):  # noqa
        self.attrs = Attributes(attrs=attrs)

    def __getitem__(self, key):
        return self.attrs[key]

    def __setitem__(self, key, value):
        self.attrs[key] = value

    def __repr__(self):
        return self.attrs.__repr__()

    def _repr_html_(self):
        return self.attrs._repr_html_()

    def fit(self):  # noqa
        raise NotImplementedError()

    def clean(self):  # noqa
        raise NotImplementedError()
