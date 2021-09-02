---
title: 'Frites: A Python package combining measures of information and group-level statistics to extract cognitive brain networks'
tags:
  - python
  - neuroscience
  - information-based
  - statistics
  - functional connectivity
  - fixed-effect ffx
  - random-effect rfx
  - cluster-based statistics
  - MEG EEG sEEG
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

The field of cognitive computational neuroscience addresses open questions regarding
the complex relation between cognitive functions and the dynamic coordination of neural
activity over large-scale and hierarchical brain networks. State-of-the-art approaches
involve the characterization of brain regions and inter-areal interactions that participate
in the cognitive process under investigation [@Battaglia:2020]. More precisely, the study of cognitive
brain networks underlies linking brain data to experimental variables, such as sensory
stimuli or behavioral responses.

Information-based measures, as developed within information theory and machine learning,
currently provide ideal tools for quantifying the link between single neural population’s
or network’s activity and task variables. Nevertheless, progress is limited by a drastically increased complexity of statistical analyses as the size and connectivity of brain
networks increases. Furthermore, identifying task-related activity from brain signals
with a poor signal-to-noise ratio represents a real challenge. 

These facts justify the need for powerful statistical approaches combined with efficient
coding strategies to quantify emerging patterns occurring during cognition from large and
complex neurophysiological datasets.

# Statement of need

[`Frites`](https://brainets.github.io/frites) (_Framework for Information
Theoretical analysis of Electrophysiological data and Statistics_) is a pure Python
package to extract cognitive brain networks by means of information-based metrics
from fields like information-theory, machine-learning or measures of distances. The
package contains statistical pipelines using permutation-based nonparametric approaches
and allow defining fixed- and random-effect models accounting for random variations
in the population. `Frites` includes a large and yet increasing set of information
measures that allow the analysis of  single-trial, dynamic, undirected (e.g., mutual
information) and directed (e.g., Granger causality) measures of functional connectivity.

`Frites` has been written for the analysis of continuous neurophysiological data,
encompassing recordings with either uniform spatial sampling (e.g., M/EEG data) and
spatially sparse intracranial recordings, such as intracranial EEG or Local Field
Potentials (LFPs). The package supports standard [`NumPy`](https://numpy.org/) array
inputs [@Harris:2020], objects from the [`MNE-Python`](https://mne.tools/stable/index.html) 
software [@Gramfort:2013], but also multi-dimensional labelled 
[`Xarray`](http://xarray.pydata.org/en/stable/) objects [@Hoyer:2017]. By default, 
task-related activity are quantified using information-theoretic measures, using a
`NumPy` tensor-based implementation of the Gaussian Copula Mutual-Information
[@Ince:2017]. The definition of custom estimators like
[`scikit-learn`](https://scikit-learn.org/stable/) cross-validated classifiers
[@Pedregosa:2011] is also supported. In addition, as some features like permutations
are computationally demanding, `Frites` natively supports parallel processing using
the [`Joblib`](https://joblib.readthedocs.io/en/latest/) Python package. Finally,
and as an optional dependency, some functions can further be accelerated using the
[`Numba`](http://numba.pydata.org/) compiler [@Lam:2015]. Programming optimizations
and external dependencies allow to investigate large-scale datasets in a reasonable
time.

Taken together, this package can be used for linking local brain activity or pairwise
links to an external variable (such as discrete stimulus types or continuous behavioral
models) and extract reproducible effects across a population. `Frites` provides
a set of high-level automated workflows that should adapt to neuroscientists with
low or moderate programming skills.


# Acknowledgements

EC and AB were supported by the PRC project “CausaL” (ANR-18-CE28-0016). This
project/research has received funding from the European Union’s Horizon 2020 Framework
Programme for Research and Innovation under the Specific Grant Agreement No. 945539
(Human Brain Project SGA3). AB was supported by FLAG ERA II  “Joint Transnational
Call 2017" - HBP - Basic and Applied Research 2, Brainsynch-Hit (ANR-17-HBPR-0001).
RB acknowledges support through a PhD Scholarship awarded by the Neuroschool. This
work has received support from the French government under the Programme Investissements
d’Avenir, Initiative d’Excellence d’Aix-Marseille Université via A\*Midex
(AMX-19-IET-004) and ANR (ANR-17-EURE-0029) funding. RAAI was supported by the Wellcome
Trust [214120/Z/18/Z]. VLC is supported by a scholarship from the European Union's
Horizon 2020 research and innovation programme under the Marie Sk lodowska-Curie
grant agreement No 859937.


# References

