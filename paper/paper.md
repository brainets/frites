---
title: 'Frites: A Python package combining measures of information and group-level statistics to extract cognitive brain networks'
tags:
  - Python
  - neuroscience
  - information
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
modulated by the task and reproducible across a population or overall, repeated
measurements.

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References

