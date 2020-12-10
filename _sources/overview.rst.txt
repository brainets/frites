Overview
========

.. contents::
   :local:
   :depth: 3


1. Goals of Frites
-------------------

Frites is an open-source Python software started in 2019 to **analyze neurophysiological data using information theoretical (IT) measures**. Overall, the goals of Frites could be summarized as follow :

1. **Extract task-related cognitive brain networks :** i.e brain regions and connectivity between brain regions that are modulated according to the task
2. **Group-level statistical inferences :** being able to draw conclusions on a population of subjects while taking into account the the richness of data at the single subject level


1.1 Task-related and feature-specificity
++++++++++++++++++++++++++++++++++++++++

The stimulus of the task (e.g visual stimulus like color circles) is first going to be processed by the brain, occurring some changes in the brain activity that are related to the stimulus. This is what we called **stimulus-specific** brain activity. Once processed, this stimulus is going to introduce some changes in the behavior of the subject (e.g decision making). Following the same reasoning as above we called **behavior-specific** the modulations in the activity of the brain that can explain a change in the behavior of the subject.

.. figure::  _static/feat_spec.png
    :align:  center

Now, with Frites we try to isolate the brain activity that is either related to the stimulus or to the behavior. In a more general way, this brain activity is called **task-related** or **feature-specific** where *feature* can either refer to the *stimulus* or to the *behavior*.

1.2 IT framework to extract task-related cognitive brain networks
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A brain network is composed of two things :

1. **Nodes :** the nodes of a network are the brain regions (e.g Insula / vmPFC / dlPFC form a three nodes network)
2. **Edges :** the connectivity links between the nodes (e.g comodulations or information flow)

Frites contains a collection of statistical methods to test whether :

1. The **local neural activity** (i.e the activity of a particular node of the network) is modulated by the task
2. The **connectivity** (which can be undirected or directed) between brain regions is also modulated by the task

.. figure::  _static/network_framework.png
    :align:  center

    Framework using information theoretical measures on neurophysiological data to extract task-related local neural activity and connectivity

2. Methods
----------

In this section we are going to cover the basis of the methods used in Frites. In particular two main topics :

1. The use of information theoretical measures for brain data
2. The implemented statistical approaches

Note that this methodological overview has been written for scientists who are not necessarily familiar with information theory and statistics. Therefore, it is not going to contain the mathematics behind the implemented methods, but instead, the references to the papers in question.

2.1 Data analysis within the information theoretical framework
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The first question that could be asked is *Why using the information theory to analyze brain data?*

There are several advantages of using information theoretic measures. Here's a non exhaustive list of the main advantages :

1. **No assumption about the distribution of the data :** standard IT measures (based on binning) are model-free in the sense that there's no assumptions about how the data are distributed
2. **Information theory as a unified framework of analysis :** most of the statistical measures we used to analyze data (e.g correlation, machine-learning etc.) have a equivalent using IT metrics. This means that in theory it is possible to analyze the data from A-Z within the IT framework
3. **Support for any data type :** by construction, IT measures support both continuous and discrete variables such as univariate and multivariate.

On the other hand, IT measures also present some disadvantages :

1. **Standard approaches usually require large data :** indeed, methods based on binning need to evaluate the probability distribution (e.g the probability that your data take values between [-1, 1]). And the larger the data, the better the estimation. In addition, binning based IT measures suffer from the curse of dimensionality.
2. **Strictly positive measures :** IT measures returned the amount of information that is shared between two variables (in bits). And this quantity is superior or equal to zero. As an example, two variables might either be correlated or anti-correlated however, in both cases the amount of information is going to be strictly positive.
3. **Computationally expansive :** depending on the implementation, the evaluation of the mutual information can be relatively slow

.. figure::  _static/gcmi_corr.png
    :align:  center

    Correlation (orange) vs. strictly positive Mutual information (purple). Extracted from **Ince et al. 2017** :cite:`ince2017statistical`

2.1.1 Gaussian Copula Mutual Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Frites we use a special type of mutual information that has been carried recently to neuroscience (see Ince et al 2017 :cite:`ince2017statistical`) : the Gaussian Copula Mutual Information (GCMI).

The GCMI provide an estimation of the mutual information **without the necessity of binning the data**. In addition to this point, the GCMI is really fast to compute which makes this method a good candidate when it comes to performing numerous permutations for the statistics. However, the GCMI make the assumption that there's a **monotonic relation between the two variables** (i.e if one variable is increasing, the other should also increase or decrease but not both).


2.1.2 The different types of mutual informations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section we are going to make the difference between two types of variables :

* **Discrete variables :** discrete variables are composed with **integers** where each number can for example refer to an experimental condition (e.g 0={all the trials that belong to the "go" stimulus} and 1={all the trials that belong to the "nogo" stimulus})
* **Continuous variables :** continuous variables are made of **floating points**. Therefore, the brain data are considered as a continuous variable.

With this in mind, the figure bellow summarizes the different types of mutual information :

.. figure::  _static/cc_cd_ccd.png
    :align:  center

    **Top left** : continuous / discrete case (``I(C; D); mi_type='cd'``) for contrasting two experimental conditions (e.g two visual stimulus); **Top right** continuous / continuous case (``I(C; C); mi_type='cc'``) for correlating the brain data with, for example, a model of the behavior (e.g Prediction Error / Learning); **Bottom** continuous / continuous | discrete case (``I(C; C | D); mi_type='ccd'``) for correlating the brain data with a continuous while removing the influence of a discrete variable (e.g experimental conditions)


2.1.3 Equivalence with other statistical measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find bellow the table showing the equivalence between information theoretic quantities and other statistical measures. Extracted from **Ince et al. 2017** :cite:`ince2017statistical`

+-------------------------------------------------------+----------------------------------------------------------------------+
| Information theoretic quantity                        | Other statistical approaches                                         |
+=======================================================+======================================================================+
| MI (discrete; discrete)                               | - Chi-square test of independence                                    |
|                                                       | - Fishers exact test                                                 |
+-------------------------------------------------------+----------------------------------------------------------------------+
| MI (univariate continuous; discrete)                  | - 2 classes: t test, KS test, Mann–Whitney U test                    |
|                                                       | - ANOVA                                                              |
+-------------------------------------------------------+----------------------------------------------------------------------+
| MI (multivariate continuous; discrete)                | - 2 classes: Hoteling T2 test                                        |
|                                                       | - Decoding (cross-validated classifier)                              |
+-------------------------------------------------------+----------------------------------------------------------------------+
| MI (univariate continuous; univariate continuous)     | - Pearson correlation                                                |
|                                                       | - Spearman rank correlation                                          |
|                                                       | - Kendall rank correlation                                           |
+-------------------------------------------------------+----------------------------------------------------------------------+
| MI (multivariate continuous; univariate continuous)   | - Generalized Linear Model framework                                 |
|                                                       | - Decoding (cross-validated regression)                              |
+-------------------------------------------------------+----------------------------------------------------------------------+
| MI (multivariate continuous; multivariate continuous) | - Canonical correlation analysis                                     |
|                                                       | - Distance correlation                                               |
+-------------------------------------------------------+----------------------------------------------------------------------+
| Conditional mutual information                        | - Partial correlation (continuous variables and linear effects only) |
+-------------------------------------------------------+----------------------------------------------------------------------+
| Directed information (transfer entropy)               | - Granger Causality                                                  |
+-------------------------------------------------------+----------------------------------------------------------------------+
| Directed feature information                          | - Dynamic Causal Modeling                                            |
|                                                       | - Psychophysiological interactions                                   |
+-------------------------------------------------------+----------------------------------------------------------------------+
| Interaction information                               | - Representational similarity analysis (redundancy only)             |
|                                                       | - Cross-classification decoding (redundancy only)                    |
|                                                       | - Mediation analysis                                                 |
+-------------------------------------------------------+----------------------------------------------------------------------+  


2.1.4 References
~~~~~~~~~~~~~~~~

* Ince et al. 2017 :cite:`ince2017statistical`
* Timme and Lapish 2018 :cite:`timme_tutorial_2018`


2.2 Statistical analyses
++++++++++++++++++++++++

In addition to the evaluation of the amount of information shared between the data and a feature (stimulus / behavior), Frites also contains a statistical pipeline to evaluate whether an effect can be considered as significant.

2.2.1 Subject and group-level statistical inferences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From a statistical standpoint, Frites gives the possibility to draw inferences either at the **subject-level** either at the **group-level** :

* **Subject-level :** the measures of informations and the p-values are computed per-subject. At the end, you can get the number (or proportion) of subjects that have the effect. This can be particularly useful when working with sEEG data where the number of subjects per brain regions might be different.
* **Group-level :** draw conclusions about a population of subjects. This group-level section is subdivided into two group-level strategies :

    * **Fixed Effect (FFX) :** the mutual information is estimated **across subjects**. By concatenating the subjects, you make the hypothesis that the effect is quite reproducible and stable across your population of subjects. One advantage of the FFX is that it usually requires a lower number of subjects and provides a good sensibility as soon as the hypothesis of stable effect across subjects is verified. On the other hand, two disadvantages are that, first you don't take into account the inter-subject variability and second, the inferences you are allow to make only concern **your population** of subjects. It can't generalize to new subjects
    * **Random Effect (RFX) :** the mutual information is estimated **per subject** and then a model of how the effect is distributed across the subjects is made. To be more precise, we make the assumption that the effect is normally distributed across the subjects. Advantages and disadvantages are basically the opposite of the FFX. The RFX takes into consideration the inter-subject variability which make it more suitable to detect the effect if this effect is slightly different from subject to subject. Building a model based is like considering that our population is in fact a sub-group of a broader population which, in other terms, means that if new subjects are included in the analysis, the estimations of mutual information should still be included in the model. However, the main disadvantage of the RFX is that it requires more subjects to build a reliable model.

.. figure::  _static/group_level.png
    :align:  center

    Subject and group-level statistics on information-based measures


2.2.2 Correction for multiple comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Internally, we used permutations in order to estimate the distribution of mutual-information values that could be obtained by chance and also, to correct the p-values for multiple comparisons. There's three ways to correct the p-values, all using `MNE Python <https://mne.tools/stable/index.html>`_ : 1) cluster-based statistics (default), 2) maximum statistics and 3) FDR. About the cluster-based, since some electrophysiological recordings don't have a spatial contiguity (like sEEG), the clusters are detected across time-points or across time-frequency maps. Note that you can also use the Threshold Free Cluster Enhancement (TFCE) still through MNE.

2.2.3 References
~~~~~~~~~~~~~~~~

* Friston et al. 1999 :cite:`friston1999many`, 1996 :cite:`friston_detecting_1996`
* Wilcox and Rousselet 2017, :cite:`wilcox_guide_2017`
* Nicohls and Holmes 2001 :cite:`nichols_nonparametric_2002`
* Maris and Oostenveld 2007 :cite:`maris2007nonparametric`
* Smith and Nichols, 2009 :cite:`smith2009threshold`
* Cao and Zhang 2014 :cite:`cao_multiple_2014`
* Combrisson et al. 2015 :cite:`combrisson_exceeding_2015`
* Giordano et al. 2017 :cite:`giordano2017contributions`

3. What you need to know to start analyzing your data with Frites
-----------------------------------------------------------------

In this section we are going to cover the basics knowledges if you want to start analyzing your data with Frites.

.. note::
    
    The gallery of examples contains a `tutorial <https://brainets.github.io/frites/auto_examples/index.html#tutorials>`_ section that explain and illustrate the main usages and features of Frites, in a notebook fashion.

3.1 Package organization
++++++++++++++++++++++++

As other Python packages, Frites contains subject-specific submodules. Here's a short description of the main submodules for new users :

* `frites.dataset <https://brainets.github.io/frites/api.html#module-frites.dataset>`_ : container for the electrophysiological data coming from multiple subjects (see also `those examples <https://brainets.github.io/frites/auto_examples/index.html#multi-subjects-dataset>`_ that explain how to define a container depending on your data type)
* `frites.workflow <https://brainets.github.io/frites/api.html#module-frites.workflow>`_ : the workflows perform a series of analyzes (usually a first analysis to extract information using IT measures follow by a second statistical analysis)
* `frites.conn <https://brainets.github.io/frites/api.html#module-frites.conn>`_ : directed and undirected connectivity metrics that can either be computed across trials or at the single trial level

In addition to the main modules above, the  `gallery of examples <https://brainets.github.io/frites/auto_examples/index.html>`_ illustrate the main functionalities of Frites using simulated data. Those functions can be imported from `frites.simulations <https://brainets.github.io/frites/api.html#module-frites.simulations>`_ and can be used to simulate local neural activity modulated by the task such as stimulus-specific brain networks using `autoregressive models <https://brainets.github.io/frites/api.html#autoregressive-model>`_.

Finally, for the developers here the `frites.core <https://brainets.github.io/frites/api.html#module-frites.core>`_ module include the very low level functions to estimate the mutual information (vector and tensor based implementations).


3.2 Main Frites' workflows
++++++++++++++++++++++++++

Frites contains two centrals workflows :

1. `WfMi <https://brainets.github.io/frites/generated/frites.workflow.WfMi.html#frites.workflow.WfMi>`_ : the main workflow of mutual-information that is used to **extract feature-specific brain networks**
2. `WfStats <https://brainets.github.io/frites/generated/frites.workflow.WfStats.html#frites.workflow.WfStats>`_ : the workflow of statistics we used to perform group-level inferences

Actually, under the hood, the **WfMi** is also using the workflow of statistics so that it return the estimation of mutual-information between a feature and the data but also the corresponding p-values.

.. warning::

    This workflow of mutual-information can be used for many things, including :

        * Provide a meaningful measure of effect-size between the data and a feature
        * To find some significant differences between multiple experimental conditions (``mi_type='cd``)
        * To correlate the brain data with a behavioral model (``mi_type='cc``) and possibly, independently of some experimental conditions (``mi_type='ccd``)
        * Get the corrected p-values to see whether you have a significant amount of information between your data and a feature, either at the group-level or at the subject level
        * Working on sparse data like the sEEG / iEEG where the number of subjects per brain region varies

    In a nutshell, the **WfMi** is the core workflow to find task-related activity. In addition, this workflow can either be used to quantify the amount of information between local neural activity and a feature **(i.e for each brain region)** but also on **connectivity links (i.e on pairs of brain regions)** to see whether there's some differences in term of connectivity between experimental conditions or ultimately, if the connectivity correlates with a behavioral model.

    To see this workflow in action, checkout `those examples <https://brainets.github.io/frites/auto_examples/index.html#mutual-information>`_


3.3 Deep integration with Xarray
++++++++++++++++++++++++++++++++

`Xarray <http://xarray.pydata.org/en/stable/>`_ is a recent python package to handle of multi-dimensional arrays using labels. For those who are familiar with `Pandas <https://pandas.pydata.org/>`_, you can see Xarray as a generalization of Pandas for multi-dimensional arrays.

Xarray provide a container for the data called `DataArray <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html#xarray.DataArray>`_. This structure comes with two important inputs : 1) `dims` which describe the name of each dimension of the array and 2) `coords` the values taken by each dimension. For example you can define a DataArray with the dimensions ``('roi', 'times')`` where ``roi = ['insula', 'vmPFC', 'dlPFC']`` and ``times = np.linspace(-3, 3, 1000)``. After that, the manipulation of the data happen using the values of the coordinates. Bellow, a minimal slicing example :

.. code-block:: python

    """
    `da` is a xarray.DataArray. With the code line below, we select the data
    coming from the two brain regions Insula and vmPFC. Then we also select
    every time points comprised between [-1.5, 1.5] seconds
    """
    da.sel(roi=['insula', 'vmPFC'], times=slice(-1.5, 1.5))


The example above only show how to slice the data but actually Xarray contains most of the operations using the same label-based syntax.

.. note::
    
    Frites make an extensive use of Xarray as most of the outputs returned are DataArrays. Since it's a relatively recent package, we wrote `two mini tutorials <https://brainets.github.io/frites/auto_examples/index.html#xarray>`_ to start working with it.
