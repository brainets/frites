"""
Estimate the empirical confidence interval
==========================================

This example illustrates how to estimate the confidence interval
"""
import numpy as np

from frites.simulations import sim_local_cc_ms
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
from frites import set_mpl_style

import matplotlib.pyplot as plt
set_mpl_style()


###############################################################################
# Plotting functions
# ------------------
#
# First, we define the function that is then going to be used for plotting the
# results

def plot(mi, pv, ci, color='C0', p=0.05, title='', units='MI (bits)'):
    # figure definition
    n_cis, n_rois = len(ci['ci']), len(mi['roi'])
    width, height = int(np.round(4 * n_rois)), int(np.round(4 * n_cis))
    fig, axs = plt.subplots(
        nrows=n_cis, ncols=n_rois, sharex=True, sharey=True,
        figsize=(width, height))
    fig.suptitle(title, fontweight='bold')

    # select significant results
    mi_s = mi.copy()
    mi_s.data[pv.data >= p] = np.nan

    # plot the results
    for n_r, r in enumerate(mi['roi'].data):
        for n_c, c in enumerate(ci['ci'].data):
            plt.sca(axs[n_c, n_r])
            plt.plot(mi['times'].data, mi.sel(roi=r).data, color='C3',
                     linestyle='--')
            plt.plot(mi['times'].data, mi_s.sel(roi=r).data, color=color, lw=3)
            plt.fill_between(
                mi['times'].data, ci.sel(ci=c, roi=r, bound='high'),
                ci.sel(ci=c, roi=r, bound='low'), alpha=.5, color=color)
            plt.title(f"ROI={r}; CI={c}%")
            plt.ylabel(units)

###############################################################################
# Data simulation
# ---------------
#
# Let's simulate some data with 10 subjects, 100 epochs per subject and 2 brain
# regions. As a result, we get a variable x representing the simulated neural
# data coming from the 10 subjects and y, the task-related variable.

n_subjects, n_epochs, n_roi = 10, 100, 2
x, y, roi, times = sim_local_cc_ms(n_subjects, n_epochs=n_epochs, n_roi=n_roi,
                                   random_state=0)
dt = DatasetEphy(x.copy(), y=y, roi=roi, times=times)

###############################################################################
# Empirical confidence interval with FFX models
# ---------------------------------------------
#
# Then, we estimate the confidence interval when using a fixed-effect model

# computes mi
wf = WfMi(mi_type='cc', inference='ffx')
mi, pv = wf.fit(dt, n_perm=200, n_jobs=1, random_state=0)

# computes confidence interval
ci = wf.confidence_interval(dt, n_boots=200, ci=[95, 99.9], n_jobs=1,
                            random_state=0)
print(ci)

# plot the results
# sphinx_gallery_thumbnail_number = 1
plot(mi, pv, ci, title='CI - FFX model')
plt.show()

###############################################################################
# Empirical confidence interval with RFX models
# ---------------------------------------------
#
# When using the random-effect model, it's either possible to estimate the
# confidence interval on the returned mutual-information or on t-values. To
# do the switch, you can use the parameter `rfx_es` for choosing between 'mi'
# or 'tvalues'

# confidence interval on mi
wf = WfMi(mi_type='cc', inference='rfx')
mi, pv = wf.fit(dt, n_perm=200, n_jobs=1, random_state=0)
ci = wf.confidence_interval(dt, n_boots=200, ci=[95, 99.9], n_jobs=1,
                            random_state=0)
plot(mi, pv, ci, title='CI - RFX model / MI')
plt.show()

###############################################################################
# confidence interval on t-values

wf = WfMi(mi_type='cc', inference='rfx')
_, pv = wf.fit(dt, n_perm=200, n_jobs=1, random_state=0)
tv = wf.tvalues
ci = wf.confidence_interval(dt, n_boots=200, ci=[95, 99.9], n_jobs=1,
                            random_state=0, rfx_es='tvalues')
plot(tv, pv, ci, title='CI - RFX model / T-values', units='T-values')
plt.show()
