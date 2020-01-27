"""Base class for worflows."""


class WfBase(object):
    """Base class for workflows."""

    def __init__(self):  # noqa
        pass

    def fit(self):  # noqa
        raise NotImplementedError()

    def clean(self):  # noqa
        raise NotImplementedError()
