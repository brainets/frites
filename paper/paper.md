---
title:  'Frites: A Python package for functional connectivity analysis and group-level statistics of neurophysiological data'
tags:
  - python
  - cognitive neuroscience
  - computational neuroscience
  - neuroinformatics
  - neurophysiology
  - information theory
  - information-based measures
  - statistics
  - functional connectivity
  - fixed-effect ffx
  - random-effect rfx
  - cluster-based statistics
  - MEG EEG sEEG LFPs
  - Granger causality
authors:
  - name: Etienne Combrisson
    orcid: 0000-0002-7362-3247
    affiliation: 1
  - name: Ruggero Basanisi
    orcid: 0000-0003-4776-596X
    affiliation: 1
  - name: Vinicius Lima Cordeiro
    orcid: 0000-0001-7115-9041
    affiliation: "1, 2"
  - name: Robin A.A Ince
    orcid: 0000-0001-8427-0507
    affiliation: 3
  - name: Andrea Brovelli
    orcid: 0000-0002-5342-1330
    affiliation: 1
affiliations:
 - name: Institut de Neurosciences de la Timone, Aix Marseille Université, UMR 7289 CNRS, 13005, Marseille, France
   index: 1
 - name: Institut de Neurosciences des Systèmes, Aix-Marseille Université, UMR 1106 Inserm, 13005, Marseille, France
   index: 2
 - name: Institute of Neuroscience and Psychology, University of Glasgow, Glasgow, UK
   index: 3
date: 01 September 2021
bibliography: paper.bib

---

# Summary
The field of cognitive computational neuroscience addresses open questions regarding the complex relation between cognitive functions and the dynamic coordination of neural activity over large-scale and hierarchical brain networks. State-of-the-art approaches involve the characterization of brain regions and inter-areal interactions that participate in cognitive processes [@Battaglia:2020]. More precisely, the study of cognitive brain networks underlies linking local neural activity or interactions between brain regions to experimental variables, such as sensory stimuli or behavioral responses. The relation between the brain data and external variables might take complex forms (e.g. non-linear relationships) with strong variations across brain regions and participants. Therefore, powerful measures of information are required to detect complex relation and the statistical relevance at the population level should be able to adapt to the inter subject variability. 

# Statement of need
[`Frites`](https://brainets.github.io/frites) (_Framework for Information Theoretical analysis of Electrophysiological data and Statistics_) is an open-source Python package, designed for researchers and students with little programming knowledge and working on the discovery of cognitive brain networks using neurophysiological data. It combines in a single framework information-based analyses of neurophysiological recordings and group-level statistical inference that can adapt to the inherent variability across participants. The measures of information and the group-level statistics functionalities are wrapped into high-level workflows. In addition, `Frites` also includes several measures of functional connectivity (FC) for measuring pairwise interactions between brain regions.

## Supported neurophysiological recordings and link with other softwares
`Frites` is suitable for the analysis of continuous and multi-channel neurophysiological data, encompassing non-invasive recordings with uniform spatial sampling (e.g., M/EEG data) such as invasive and spatially sparse recordings, such as intracranial EEG or Local Field Potentials (LFPs). The package supports standard [`NumPy`](https://numpy.org/) array inputs [@Harris:2020], neuro-oriented objects from the [`MNE-Python`](https://mne.tools/stable/index.html) software [@Gramfort:2013], but also multi-dimensional labelled  [`Xarray`](http://xarray.pydata.org/en/stable/) objects [@Hoyer:2017].

## Measures of information and group-level statistics
`Frites` is equipped with a set of information theoretic tools for the analysis of interactions between brain signals and their relation with experimental task-related variables. By default, `Frites` is using the Gaussian Copula Mutual-Information [@Ince:2017] to study the relation between either local brain activity, cross-frequency coupling [@Combrisson:2020] or inter-areal FC with experimental variables (i.e., cognitive tasks). For what concerns FC, the toolbox allows the estimate of dynamic (i.e., time-resolve) [@Brovelli:2017], undirected (e.g., mutual information) and directed (e.g., Granger causality) FC on a single-trial basis [@Brovelli:2015]. Nevertheless, the definition of custom estimators, such as kernel methods [@Wollstadt:2019] and [`scikit-learn`](https://scikit-learn.org/stable/) cross-validated classifiers [@Pedregosa:2011], is also supported.

For statistical inferences, the package integrates a non-parametric permutation-based statistical framework [@Maris:2007] to perform group-level inferences on non-negative measures of information. The toolbox includes different methods that cope with multiple-comparison correction problems, such as test- and cluster-wise p-value corrections. The implemented framework also supports both fixed- and random-effect models to adapt to inter-individuals and inter-sessions variability [@Combrisson:2022].

## Workflows
`Frites` provides a set of workflows that integrate several analysis steps. Those workflows take as inputs the neural data coming from single or multi-participants (or single / multi sessions), estimate the amount of information shared between the brain data and the external variable, at each brain region and time bins, and finally perform network-level statistical inference, corrected for multiple comparisons. Those high-level workflows are particularly useful for users with no or little programming knowledge as all of those steps can be performed in a very few lines of code, with standard inputs like `MNE-Python` epochs.

## Performance
Several computations implemented in the workflows, such as permutation tests, are computationally demanding. To decrease computing time, the core functions for information measures exploit a `NumPy` tensor-based implementation or can be accelerated using the [`Numba`](http://numba.pydata.org/) compiler [@Lam:2015]. Then, `Frites` natively supports parallel processing using the [`Joblib`](https://joblib.readthedocs.io/en/latest/) package. 

# Acknowledgements
EC and AB were supported by the PRC project “CausaL” (ANR-18-CE28-0016). This project/research has received funding from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project SGA3). RB acknowledges support through a PhD Scholarship awarded by the Neuroschool. This work has received support from the French government under the Programme Investissements d’Avenir, Initiative d’Excellence d’Aix-Marseille Université via A\*Midex (AMX-19-IET-004) and ANR (ANR-17-EURE-0029) funding. RAAI was supported by the Wellcome Trust [214120/Z/18/Z]. VLC is supported by a scholarship from the European Union's Horizon 2020 research and innovation programme under the Marie Sk lodowska-Curie grant agreement No 859937.

# References
