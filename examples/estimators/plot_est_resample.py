"""
Trial-resampling: correcting for unbalanced designs
===================================================

This example illustrates how to correct information estimation in case of
unbalanced designs (i.e. when the number of epochs or trials is very different
between conditions).

The technique of trial-resampling consist in randomly taking an equal number of
trials per condition, estimating the effect size and then repeating this
procedure for a more reliable estimation.
"""
import numpy as np
import pandas as pd

from frites.estimator import GCMIEstimator, ResamplingEstimator, DcorrEstimator
from frites import set_mpl_style

import seaborn as sns
import matplotlib.pyplot as plt
set_mpl_style()


###############################################################################
# Data creation
# -------------
#
# This first section creates the data using random points drawn from gaussian
# distributions

n_variables = 1000  # number of random variables
n_epochs = 500      # total number of epochs
prop = 5            # proportion (in percent) of epochs in the first condition

# proportion of trials
n_prop = int(np.round(prop * n_epochs / 100))

# create continuous variables
x_1 = np.random.normal(loc=1., size=(n_variables, 1, n_prop))
x_2 = np.random.normal(loc=2., size=(n_variables, 1, n_epochs - n_prop))
x = np.concatenate((x_1, x_2), axis=-1)
y_c = np.r_[np.random.normal(size=(n_prop,)),
            np.random.normal(size=(n_epochs - n_prop,))]

# create discret variable
y_d = np.array([0] * n_prop + [1] * (n_epochs - n_prop))

print(f"Smaller dataset : {x_1.shape}")
print(f"Larger dataset : {x_2.shape}")


###############################################################################
# Information shared between a continuous and a discret variable
# --------------------------------------------------------------
#
# In this second section, we define an estimator for computing the information
# shared between a continuous and a discret variable. In a second step, we are
# going to wrap this estimator with a trial-resampling estimator.

# mutual information uncorrected estimator
est = GCMIEstimator(mi_type='cd', biascorrect=False)
mi_1 = est.estimate(x, y_d).squeeze()

# mutual information corrected estimator (with trial-resampling)
est_r = ResamplingEstimator(est, n_resampling=100)
mi_2 = est_r.estimate(x, y_d).squeeze()

df = pd.DataFrame({
    'MI': np.r_[mi_1, mi_2],
    'Estimator': ['Uncorrected'] * len(mi_1) + ['Corrected'] * len(mi_2)
})

###############################################################################
# .. note::
#     As shown below, the effect size for the corrected estimator is slightly
#     over the non-corrected one.

sns.displot(df, x='MI', hue='Estimator', kde=True, height=7)
plt.title("Information shared between a continuous and a discrete variable")
plt.tight_layout()
plt.show()


###############################################################################
# Information shared between two continuous variables
# ---------------------------------------------------
#
# In this last section, we define an estimator for computing the information
# shared between two continuous variables. Similarly to above, we are then
# going to wrap this estimator with a trial-resampling estimator.

# distance correlation uncorrected estimator
est = DcorrEstimator()
mi_1 = est.estimate(x, y_c, z=y_d).squeeze()

# distance correlation corrected estimator (with trial-resampling)
est_r = ResamplingEstimator(est, n_resampling=20)
mi_2 = est_r.estimate(x, y_c, z=y_d).squeeze()

df = pd.DataFrame({
    'MI': np.r_[mi_1, mi_2],
    'Estimator': ['Uncorrected'] * len(mi_1) + ['Corrected'] * len(mi_2)
})

###############################################################################
# .. note::
#     As shown below, the effect size for the corrected estimator is slightly
#     over the non-corrected one.

sns.displot(df, x='MI', hue='Estimator', kde=True, height=7)
plt.title("Information shared between two continuous variables")
plt.tight_layout()
plt.show()