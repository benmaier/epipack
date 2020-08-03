# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [v0.0.0] - 2020-06-22
### Changed
- initialized

[Unreleased]: https://github.com/benmaier/epipack/compare/v0.0.4...HEAD
[v0.0.4]: https://github.com/benmaier/epipack/compare/v0.0.3...v0.0.4]
[v0.0.3]: https://github.com/benmaier/epipack/compare/v0.0.2...v0.0.3]
[v0.0.2]: https://github.com/benmaier/epipack/compare/v0.0.1...v0.0.2]
[v0.0.1]: https://github.com/benmaier/epipack/compare/v0.0.0...v0.0.1]
