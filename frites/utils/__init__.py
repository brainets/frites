"""Global utility functions."""
from .parallel import parallel_func  # noqa
from .preproc import (  # noqa
    savgol_filter, kernel_smoothing, nonsorted_unique, time_to_sample,
    get_closest_sample
)
from .wrapper import jit  # noqa
