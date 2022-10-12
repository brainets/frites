"""
Define temporal windows
=======================

This example illustrates how to define temporal windows that can then be used
to compute, for example, the dynamic functional connectivity. In this example,
we show how to define either custom windows or sliding windows with different
sizes.
"""
import numpy as np

from frites.conn import define_windows, plot_windows
from frites import set_mpl_style

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
set_mpl_style()


###############################################################################
# Simulate data
# -------------
#
# Let's start by creating a simple sinusoide with a time vector. Then, to
# define windows we are going to use the function
# :func:`frites.conn.define_windows` and also
# :func:`frites.conn.plot_windows` for plotting.

n_pts = 1000
period = .2  # 200ms period
times = np.linspace(-1, 1.5, n_pts, endpoint=True)
x = np.sin(2 * np.pi * (1. / period) * times)


###############################################################################
# Manually define windows
# -----------------------
#
# The first feature we illustrate here, is to define windows manually which
# means, based on the time vector, where each window should start and finish.

plt.figure(figsize=(10, 8))

# by default, if you don't provide any input, the full time window is going to
# be considered
win = define_windows(times)[0]
plt.subplot(311)
plot_windows(times, win, x, title='No input = full time window')

# however, you can also specify where each window start and finish. Here, we
# define a simple unique window between [-.5, .5]
win = define_windows(times, windows=[-.5, .5])[0]
plt.subplot(312)
plot_windows(times, win, x, title='Unique window [-.5, .5]')

# but you can also specify multiple windows
win = define_windows(times, windows=[[-.75, -.25], [.25, .75]])[0]
plt.subplot(313)
plot_windows(times, win, x, title='Multiple windows manually defined')
plt.tight_layout()
plt.show()


###############################################################################
# Define sliding windows
# ----------------------
#
# You can also define sliding windows which can be controlled through four
# parameters :
#
# * `slwin_len` = window length
# * `slwin_start` = starting time point
# * `slwin_stop` = stopping time point
# * `slwin_step` = step time between consecutive windows

# sphinx_gallery_thumbnail_number = 2
plt.figure(figsize=(10, 10))

# if you only gives the length, consecutive windows are defined from first to
# last time point
win = define_windows(times, slwin_len=.2)[0]
plt.subplot(411)
plot_windows(times, win, x, title='200ms consecutive windows')

# but you can also change where the sliding window start
win = define_windows(times, slwin_len=.2, slwin_start=-.5)[0]
plt.subplot(412)
plot_windows(times, win, x, title='Start sliding windows from -.5s')

# in addition, you can also define where it stop
win = define_windows(times, slwin_len=.2, slwin_start=-.5, slwin_stop=.52)[0]
plt.subplot(413)
plot_windows(times, win, x,
             title='Start sliding windows from -.5s and finish at .5s')

# finally, we can control the "distance" between each consecutive window. For
# example, if the length is 200ms and a distance of 150ms, it is equivalent to
# a 25% overlap
win = define_windows(times, slwin_len=.2, slwin_start=-.75, slwin_stop=1,
                     slwin_step=.15)[0]
plt.subplot(414)
plot_windows(times, win, x,
             title='Sliding windows from [-.75, 1]s with 25% overlap')

plt.tight_layout()

plt.show()