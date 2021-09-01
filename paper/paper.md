---
title: 'Frites: A Python package combining measures of information and group-level statistics to extract cognitive brain networks'
tags:
  - Python
  - neuroscience
  - information-based
  - statistics
  - functional connectivity
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

Brain activity is consistently changing but reproducible perturbations can be
triggered either by external stimuli, repetitive behavior or can even be present
in absence of any stimulation. The field of computational neuroscience aims to
extract and quantify emerging patterns occurring during cognition, understand
how information is locally processed such as how brain regions coordinate.
That said, approaching the brain as a deeply connected network drastically
increase the complexity of statistical analyses. Over and above, identifying
task-related activity in the context of noisy measurements and therefore, a poor
signal-to-noise ratio is a real challenge. This situation justify the use of
powerful statistical approaches linking neurophysiological data to an external
variable combined to sophisticated group-level statistics to extract brain activity
modulated by the task and reproducible effects across a population or overall,
repeated measurements.

# Statement of need

[`Frites`](https://brainets.github.io/frites) (_Framework for Information
Theoretical analysis of Electrophysiological data and Statistics_) is a pure Python
package to extract cognitive brain networks by means of information-based metrics
from fields like information-theory, machine-learning or measures of distances. The
package also contains statistical pipelines using permutation-based nonparametric
approaches and allow defining fixed- and random-effect models accounting for random
variations in the population. In addition, `Frites` further includes single-trial,
dynamic, undirected and directed measures of functional connectivity.

`Frites` has been written for the analysis of electrophysiological data, encompassing
both recordings with uniform spatial sampling like M/EEG data and spatially sparse
intracranial recordings. The package supports standard `NumPy` array inputs [@Harris:2020],
objects from the MNE-Python software [@Gramfort:2013] but also multi-dimensional labelled
`Xarray` objects [@Hoyer:2017]. By default, task-related activity are quantified using information-theoretic measures, using a `NumPy` tensor-based implementation of the
Gaussian Copula Mutual-Information [@Ince:2017]. That being said, the definition of
custom estimators like `scikit-learn` cross-validated classifiers [@Pedregosa:2011]
are also supported. In addition, as some features like permutations are computationally
demanding, `Frites` natively supports parallel processing using the `Joblib` Python
package. Finally, and as an optional dependency, some functions can further be
accelerated using the `Numba` compiler [@Lam:2015]. Programming optimizations
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
(AMX-19-IET-004) and ANR (ANR-17-EURE-0029) funding.

# References

