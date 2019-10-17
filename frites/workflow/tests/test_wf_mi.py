"""Test workflow of mutual information."""
import numpy as np

from frites.workflow import WfMi


class WfMi(object):  # noqa

    def test_definition(self):
        """Test workflow definition."""
        WfMi('cc', 'rfx')
