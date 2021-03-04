---
title: 'epipack: An infectious disease modeling suite for Python'
tags:
  - Python
  - infectious disease modeling
  - stochastic simulations
  - computer algebra systems
  - networks
  - visualization
authors:
  - name: Benjamin F. Maier
    orcid: 0000-0001-7414-8823
    affiliation: "1, 2"
affiliations:
 - name: Institute for Theoretical Biology, Humboldt-University of Berlin, Philippstr. 13, D-10115 Berlin
   index: 1
 - name: Robert Koch Institute, Nordufer 20, D-13353 Berlin
   index: 2
date: 3 March 2021
bibliography: paper.bib
---

# Summary

Analyzing the spread of infectious diseases by means of compartmental mathematical models
has been an active area of research for almost a century
[@kermack_contributions_1991,@keeling_modeling_2011,@anderson_infectious_2010].
Since the emergence of the coronavirus disease 2019 pandemic in early 2020, 
the field has seen yet another considerable boost in interest. Researchers
have since been working on a
large number of different models that are used to forecast case numbers,
to analyze the implications of different contact structures between individual people, or to discuss
the influence of various mitigation and containment strategies, to only name a few applications
[@estrada_covid-19_2020].

The complexity of detailed epidemiological models often mitigates simple replication
and/or adaption to local circumstances.
Typically, modifications are quickly thought up, but their influence on system properties such as the 
epidemic threshold or outbreak size are less clear.
Variations in model formulation often entail the
reimplementation of simulation and analysis frameworks for every model iteration,
taking up valuable time and resources for debugging and reanalysis.
Furthermore, researchers often need to cross-check their results by implementing both
deterministic well-mixed models as well as models that consider explicit contact structures
(i. e. static or temporal networks). Last but not least, analytical derivations are often done using separate computer algebra systems.

*epipack* solves the raised issues by offering a simple, process-based
framework that allows researchers to quickly prototype Poisson-based compartmental epidemiological models
and to investigate their behavior based on analytical, numerical, stochastical,
and agent-based/network-based simulations, facilitated by a visualization framework and
parsimonious, customizable interactives.

Here, the overarching design principle focuses on defining epidemiological models via
reaction processes or reaction events, from which ODEs and simulation classes are generated
automatically. This allows the user to transfer implemented models quickly between analytical,
numerical, or stochastical formulations.

*epipack* provides four base classes to accomodate building models for different analysis methods:

* EpiModel: Define a model based on transition, birth, 
  death, fission, fusion, or transmission reactions for integrating the 
  ordinary differential equations (ODEs) of the corresponding well-mixed system
  numerically or simulate the system using Gillespie's algorithm [@gillespie_exact_1977].
  Process rates can be numerical functions of time and the system state.
* SymbolicEpiModel: Define a model based on transition, birth, 
  death, fission, fusion, or transmission reactions. Obtain the ODEs,
  fixed points, Jacobian, and the Jacobian's eigenvalues at fixed points
  as symbolic expressions using sympy [@meurer_sympy_2017]. 
  Process rates can be symbolic expressions of time and the system state.
  Set numerical parameter values and integrate the ODEs numerically, or
  simulate the stochastic systems using Gillespie's algorithm [@gillespie_exact_1977].
* StochasticEpiModel: Define a model based on node transition and
  link transmission reactions. Add conditional link transmission reactions.
  Simulate your model on any (un-/)directed, (un-/)weighted static/temporal
  network, or in a well-mixed system. We make use of a generalized
  version of the tree-based rejection sampling algorithm recently proposed [@st-onge-efficient_2019]
  and the accompanying implementation of *SamplableSet* [@st-onge_samplableset_nodate]. These
  algorithms are based on exact continuous-time simulations, as 
  discrete-time approximative simulation algorithms like the individual-based update algorithm
  are known to behave problematically at times
  [@givan_predicting_2011, @maier_spreading_2020, @kiss_mathematics_2017].
  The model further allows to define chained (i. e. conditional) reactions 
  using which public health interventions that resemble contact tracing can be simulated.
  This class is comparable to the `Gillespie_simple_contagion` function of the `EoN` (Epidemics on Networks) package [@miller_eon_2019], which, however, does not yet support temporal networks or conditional reactions.
* MatrixEpiModel: A static-rate version of the EpiModel class that runs faster
  on complex models (e. g. reaction-diffusion systems) by making use
  of scipy's implementation of sparse matrices [@virtanen_scipy_2020].

Moreover, we provide a simple OpenGL-based visualization framework to animate
stochastic simulations on networks, lattices, well-mixed systems,
or reaction-diffusion systems.
The research process is further
facilitated by interactive analysis widgets for *Jupyter* notebooks
that give immediate visual feedback
regarding a system's inner workings.

While other reaction-based modeling packages exist, most focus either purely on ODE systems (e. g. *ChemPy* [@dahlgren_chempy_2018]) or simulations and analyses on static network systems (for instance *EoN* (Epidemics on Networks) [@miller_eon_2019] or *epydemic* [@dobson_epydemic_2017]). One exception is the *EpiModel* package, which is, however, only available for the R language. 

*epipack* and its usage is exhaustively documented, with the documentation available at [epipack.benmaier.org](http://epipack.benmaier.org) and in the repository.

To the best of our knowledge, *epipack* is the first open source software suite for Python
that offers extensive model building and analysis frameworks for both mean-field and networks models with a simple and intuitive API.
It thus presents a valuable tool for researchers
in the infectious diseases modeling community.

![Example use cases of *epipack*.](Fig1.png)

# Acknowledgments

BFM is financially supported as an *Add-On Fellow for Interdisciplinary Life Science* by the Joachim Herz Stiftung. BFM wants to thank A. Burdinski and K. Ledebur for valuable feedback as well as J. S. Juul for the productive collaboration on a code base that heavily inspired this project.

# References
