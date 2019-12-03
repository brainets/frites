"""Base class for worflows."""
from frites.stats import STAT_FUN

class WfBase(object):
    """Base class for workflows."""

    def __init__(self):  # noqa
        pass

    def fit(self):  # noqa
        raise NotImplementedError()

    def clean(self):  # noqa
        raise NotImplementedError()

    def _check_stat(self, inference, stat_method):
        """Check that the statistical method does exist."""
        try:
            is_string = isinstance(stat_method, str)
            if is_string:
                STAT_FUN[inference][stat_method]
        except KeyError:
            m_names = [k.__name__ for k in STAT_FUN[inference].values()]
            raise KeyError(
                f"Selected statistical method `{stat_method}` doesn't "
                f"exist. For {inference} inference, use either : "
                f"{', '.join(m_names)}")
        return is_string

