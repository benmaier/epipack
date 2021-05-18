# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [v0.1.4] - 2020-05-18

### Added

- A new small-world network styling method based on 1d lattice distance (in epipack.networks)
- methods to compute Jacobian and next generation matrices (NGMs) in `MatrixEpiModel`, as well as R0 from said NGMs (TODO: add docs)
- tests for these methods
- `epipack.distributions` module, which deals with fitting empirical distributions to sums of exponentially distributed random variables (still in dev mode, also TODO: add docs)
- tests for this module
- methods to `EpiModel` that save events that have been set. This will be used to generate model flowcharts with graphviz at some point
- the possibility to pass a function to ``StochasticEpiModel.simulate`` that checks for a custom stop condition

## [v0.1.3] - 2020-04-07

### Fixed

- dependency issues with pyglet, apparently the "shapes" module did not appear until lately. Defined a range of versions for pyglet
- bug in example code in README.md

## [v0.1.2] - 2020-04-01

### Fixed

- A bug where the `reset_events`-flag was ignored when setting processes

## [v0.1.1] - 2020-03-03

### Added

- GeneralInteractiveWidget was added to allow interactive display of general functions
- a very basic SDE integrator was added (no diffusion coefficent matrix, and no system-dependent diffusion coefficients)

### Changed

- InteractiveIntegrator can now plot derivatives
- Range and LogRange classes will behave like floats whenever necessary

### Fixed

- behavior of the SamplableSet class
- a bug where the reaction rate of nodes in weighted networks is scaled by the node's degree and not by its strength

## [v0.1.0] - 2020-10-21

### Added
- `epipack.interactive`: contains a class that adds an interactive widget to Jupyter notebooks
  with which one may control the parameter values of a SymbolicEpiModel instance
- `epipack.temporal_networks`: set up temporal networks and model simulations on them
- `SymbolicODEModel`: A model that's defined via ODEs in sympy format.

## [v0.0.5] - 2020-08-14

### Changed
- DeterministicEpiModel is now MatrixEpiModel
- SymbolicEpiModel is now SymbolicMatrixEpiModel
- in StochasticEpiModel and during visualization, a more efficient mechanism checks for whether the simulation has ended for good

### Added
- Added models that are based entirely on events. In this way, we can easily implement time-dependent rates and have single models that can do everything at once: symbolic evaluations, numerical evaluations, and stochastic mean-field simulations
- time-dependent rates are integrated using a time-varying Gillespie algorithm
- EpiModel, StochasticSIModel, StochasticSIRModel, StochasticSISModel, SymbolicEpiModel (based on events rather than rates)

## [v0.0.4] - 2020-08-03
### Changed
- SymbolicEpiModel: raise error when `disease_free_state` is not given explicitly and no S-compartment can be found
- allow non-unity initial conditions for SymbolicEpiModel and DeterministicEpiModel
- `population_size` is now explicitly regarded in SymbolicEpiModel
- in DeterministicEpiModel, instead of raising errors, warnings are raised for nonzero column sums
- in StochasticEpiModel, save the current state after the end of the simulation

### Added
- A complete visualization framework and network grid layout
- in StochasticEpiModel, a callback function can be passed that's called whenever a sample is taken during the simimulation

### Fixed
- fixed bug where fission processes were converted to quadratic rates

## [v0.0.3] - 2020-06-30
### Added
- Catch situations where the true total event is zero but the maximum total event rate is non-zero

## [v0.0.2] - 2020-06-29
### Changed
- Catching ModuleNotFoundError properly

## [v0.0.1] - 2020-06-25
### Added
- Working package

## v0.0.0 - 2020-06-22
### Changed
- initialized

[Unreleased]: https://github.com/benmaier/epipack/compare/v0.1.4...HEAD
[v0.1.4]: https://github.com/benmaier/epipack/compare/v0.1.2...v0.1.4]
[v0.1.3]: https://github.com/benmaier/epipack/compare/v0.1.2...v0.1.3]
[v0.1.2]: https://github.com/benmaier/epipack/compare/v0.1.1...v0.1.2]
[v0.1.1]: https://github.com/benmaier/epipack/compare/v0.1.0...v0.1.1]
[v0.1.0]: https://github.com/benmaier/epipack/compare/v0.0.5...v0.1.0]
[v0.0.4]: https://github.com/benmaier/epipack/compare/v0.0.4...v0.0.5]
[v0.0.4]: https://github.com/benmaier/epipack/compare/v0.0.3...v0.0.4]
[v0.0.3]: https://github.com/benmaier/epipack/compare/v0.0.2...v0.0.3]
[v0.0.2]: https://github.com/benmaier/epipack/compare/v0.0.1...v0.0.2]
[v0.0.1]: https://github.com/benmaier/epipack/compare/v0.0.0...v0.0.1]
