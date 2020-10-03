# Outlook

A collection of ideas that need could be implemented.

## Approximations of temporal rates

If rates have an explicit temporal dependency, the exact method 
for stochastic simulations solves an integral numerically.

We should introduce a method in
`EpiModel.get_time_leap_and_proposed_compartment_changes`
that approximates rates as either 

1. constant. Just sample a time leap from an exponential distribution
    based on the current rates. Re-evaluate rates at the new time
    and choose event from those. 
2. linear. Sample a time leap from an exponential distribution, evaluate
    rates at the new time. Assume linear interpolation and solve this
    integral for a new time leap. Re-evaluate rates at the new time
    and choose event from those.

## Simulations on temporal networks

Straight-forward to set up based on
[tacoma](https://github.com/benmaier/tacoma)-types.

## Spatial systems

Based on our implementation in the `metapop`-prototype (github.com/benmaier/metapop), it should be fairly simple to set up more efficient reaction-diffusion systems for `MatrixEpiModel`.

## Fitting to data

- using `lmfit`, we could implement a concise fitting tool to fit models to data
- or pymc3: https://docs.pymc.io/notebooks/ODE_API_introduction.html#Non-linear-Differential-Equations
- https://docs.pymc.io/notebooks/ODE_with_manual_gradients.html
