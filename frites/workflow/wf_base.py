"""Base class for worflows."""


class WfBase(object):
    """Base class for workflows."""

    def __init__(self):  # noqa
        self.cfg = dict()

    def __getitem__(self, key):
        return self.cfg[key]

    def __setitem__(self, key, value):
        self.cfg[key] = value

    def _attrs_xarray(self, dat):
        """Set attributes to a DataArray or Dataset."""
        for k, v in self.cfg.items():
            dat.attrs[k] = 'none' if v is None else v
        return dat

    def update_cfg(self, **kwargs):
        for k, v in kwargs.items():
            self.cfg[k] = v

    def fit(self):  # noqa
        raise NotImplementedError()

    def clean(self):  # noqa
        raise NotImplementedError()
