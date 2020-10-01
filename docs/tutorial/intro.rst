Introduction
============

Since the emergence of the COVID-19 pandemic in early 2020,
epidemiological modelling has seen a considerable boost in interest.
Researchers, journalists, and laymans alike have since been working on
a large number of different models that are supposed to either
predict case numbers on a short time scale or to discuss the implications
of various containment/mitigation scenarios for longer time frames.

The complexity of such models often mitigates simple replication and/or
adaption to local circumstances. Typically, such modifications are quickly
thought up, but their influence on system properties such as the epidemic threshold
are less clear, hence new evaluations have to be performed and
new numerical solutions have to be implemented for every model iteration.

`epipack` aims at solving this issue by providing a simple, 
process-based framework to quickly 
prototype any compartmental epidemiological model and to investigate
their implications based on analytical, numerical, stochastical, and
agent-based or network-based simulations. The research process is
further facilitated by a one-size-fits-all visualization framework
and interactive analysis routines that give immediate visual feedback
regarding a system's inner workings.

In this documentation, we will define general Markovian compartmental
models and demonstrate how specific models can be constructed and 
studied using `epipack`.

Constant Rates
--------------

Simple epidemiological models are usually based on the assumption
that two individuals, an infected (`I`) and a susceptible (`S`),
can interact in a way that transmits the infection from the infected to the
susceptible individual. In a no-memory (Markovian) picture, such a process
can be defined by means of chemical reactions, such as

.. math::

    S + I \stackrel{\alpha}{\longrightarrow} I + I

which formalizes what we said above: a contact between an `S` particle
(individual) and an `I` particle (individual) leads to the decay of the
`S` particle to an `I` particle. This reaction takes place with 
rate :math:`\alpha > 0`.

Additionally, individuals are assumed to spontaneously recover, i.e.
an `I` particle decays to an `R` particle with rate :math:`\beta`,
such that

.. math::

    I \stackrel{\beta}{\longrightarrow} R.

`S`, `I`, and `R` are usually referred to as `compartments`
which individuals can be part of, which means that `S` quantifies
the number (or fraction, depending on the definition) of susceptibles
in the population, etc.

Sometimes, models also assume that previously non-existent 
susceptible individuals
are born with constant rate :math:`\gamma` while individuals
from 

.. math::

    \varnothing \stackrel{\gamma}{\longrightarrow} S.

These reaction equations can be formalized in a system of
ODEs

.. math::

    \frac{d}{dt}S &= -\alpha SI/N + \gamma\\
    \frac{d}{dt}I &= \alpha SI/N - \beta I\\
    \frac{d}{dt}R &= \beta I\\
    N &= S + I + R.

Note that this is a system of constantly growing population size
which is an usual situation but for the sake of having a non-trivial
example we will keep it like that.

The ODEs above could now be implemented in an ODE solver and 
numerically integrated or  investigated analytically with either 
pen and paper or computer algebra systems.

While reaction equations such as the ones above are often used,
specific situations may need careful adjustment of the basic reactions,
e.g. by introducing compartments for asymptomatic infectious or
compartments for quarantined individuals. Such adaptions would then
have to be translated into new ODE systems which takes time to set up and
debug. In principle, the
following reaction equations describe all possible epidemiological models

.. math::

    Y_i + Y_j &\stackrel{\alpha_{ijk\ell}}{\longrightarrow} Y_k + Y_\ell\\
    Y_i &\stackrel{\beta_{ij}}{\longrightarrow} Y_j

where :math:`Y_i` is any of :math:`C` compartments and 
:math:`Y_i = \varnothing` is a valid choice in any of the reaction
equations. These reaction equations are still somewhat restrictive,
considering that in odd cases, asymmetric reactions are mathematically
allowed. For constant rates, a generalized deterministic epidemiological model
can therefor be defined based on a system of second-order coupled 
ordinary differential equations (ODEs)

.. math::
    
    \frac{d}{dt}Y_i = \sum_{j,k} \alpha_{ijk} Y_jY_k/N + \sum_j \beta_{ij} Y_j + \gamma_i

where the population of size `N` is assumed to be sorted into
`C` compartments :math:`Y_1, Y_2, ..., Y_C` such that

.. math::
    
    N = \sum_{i=1}^C Y_i.

`epipack` allows one to set up systems like these explicitly.
For constant rates, one may use
:class:`epipack.numeric_epi_models.EpiModel`,
:class:`epipack.symbolic_epi_models.SymbolicEpiModel`,
:class:`epipack.numeric_matrix_epi_models.MatrixEpiModel`,
or
:class:`epipack.symbolic_matrix_epi_models.SymbolicMatrixEpiModel`.

While `EpiModel` and `SymbolicEpiModel` are event-based implementations
that allow for stochastic mean-field simulations, too, they can be slow
to set up and to run for increasingly complex systems. Hence, if you're
dealing with 
constant-rate systems of a large number of compartments/couplings, you
may fall back to `MatrixEpiModel` or `SymbolicMatrixEpiModel` which
are defined based on sparse matrix implementations and therefore faster
to both set up and for numeric integrations. Yet, mean-field stochastic
simulations only work with the first two base models.

Functional Rates
----------------

In general, we do not have to assume that rates are constant. They
can depend both on the current system state as well as on time explicitly.

The generalized Markovian system therefore reads

.. math::

    \frac{d}{dt}Y_i = \sum_{j,k} \alpha_{ijk}(t,Y_1,Y_2,...) Y_jY_k/N + 
                      \sum_{j} \beta_{ij}(t,Y_1,Y_2,...\}) Y_j + 
                      \gamma_i(t,Y_1,Y_2,...).

Such systems can be set up and analyzed analytically, numerically, or
based on mean-field stochastic simulations with
:class:`epipack.numeric_epi_models.EpiModel`,
:class:`epipack.symbolic_epi_models.SymbolicEpiModel`.
