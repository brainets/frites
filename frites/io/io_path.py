"""I/O functions."""
from pkg_resources import resource_filename
import pandas as pd


def get_data_path(file=None):
    """Get the path to brainets/data/.

    Alternatively, this function can also be used to load a file inside the
    data folder.

    Parameters
    ----------
    file : str
        File name

    Returns
    -------
    path : str
        Path to the data folder if file is None otherwise the path to the
        provided file.
    """
    file = file if isinstance(file, str) else ''
    return resource_filename('brainets', 'data/%s' % file)


def load_marsatlas():
    """Get the MarsAtlas dataframe.

    MarsAtlas parcels are described here [1]_.

    Returns
    -------
    df : DataFrame
        The MarsAtlas as a pandas DataFrame

    References
    ----------
    .. [1] Auzias, G., Coulon, O., & Brovelli, A. (2016). MarsAtlas: a cortical
       parcellation atlas for functional mapping. Human brain mapping, 37(4),
       1573-1592.
    """
    ma_path = get_data_path('MarsAtlas_2015.xls')
    df = pd.read_excel(ma_path).iloc[:-1]
    df["LR_Name"] = df["Hemisphere"].map(str) + ['_'] * len(df) + df["Name"]
    return df
