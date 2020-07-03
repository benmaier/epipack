# Outlook

A collection of ideas that need implementation.

## Spatial Systems

Based on our implementation in the `metapop`-prototype (github.com/benmaier/metapop), it should be fairly simple to set up reaction-diffusion systems for `DeterministicEpiModel`.

## Functional rates

- functional rates are implemented for `SymbolicEpiModel` but should be made possible for `DeterministicEpiModel` as well.
- rates should be allowed to be functions of time as well as the system state

## Copy models from other models

- Say you have set up a `SymbolicEpiModel`. You should be able to just integrate the ODEs without the hassle of copy-pasting everything you've done to create a `DeterministicEpiModel`.

## Fitting to data

- using `lmfit`, we could implement a concise fitting tool to fit models to data
