"""Base estimator for the mutual information."""
from frites.io import Attributes
from frites.config import CONFIG

from frites.io import set_log_level, logger


class BaseMIEstimator(object):

    """Base class for mutual-information estimators.

    Parameters
    ----------
    mi_type : {'cc', 'cd', 'ccd', 'ccc'}
        Mutual information type :

            * 'cc' : MI between two continuous variables
            * 'cd' : MI between a continuous and a discret variables
            * 'ccd' : MI between two continuous variables conditioned by a
              third discret one
            * 'ccc' : MI between two continuous variables conditioned by a
              third continuous one
    add_str : string | ''
        Additional string
    """

    def __init__(self, mi_type='cc', add_str='', verbose=None):
        """Init."""
        set_log_level(verbose)
        desc = CONFIG['MI_REPR'][mi_type]
        settings = {'description': desc}
        self.settings = Attributes(attrs=settings, section_name='Settings')
        self._kwargs = dict()
        assert hasattr(self, 'name')

        logger.info(f"{self.name} (mi_type={mi_type}{add_str})")

    def __repr__(self):
        """Overall representation."""
        return '*** ' + self.name + ' ***\n' + self.settings.__repr__()

    def _repr_html_(self):
        """IPython representation."""
        title = f"<h3><br>{self.name}</br></h3>"
        return title + self.settings._repr_html_()

    def estimate(self, x, y, z=None, categories=None):
        """Estimate the (possibly conditional) mutual-information."""
        raise NotImplementedError()

    def get_function(self):
        """Get the function to execute.

        The returned function should have the following signature :

            * fcn(x, y, z=None, categories=None)

        and should returned an array of shape (n_categories, n_var).
        """
        raise NotImplementedError()
