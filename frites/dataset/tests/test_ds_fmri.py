"""Test fMRI dataset."""
from frites.dataset import DatasetFMRI


class TestDatasetFMRI(object):  # noqa

    def test_definition(self):
        """Test the definition."""
        DatasetFMRI()
