.. _dev-gillespie:

Gillespie's SSA
---------------

At several points, `epipack` allows to run stochastic simulations
which are all based on Gillespie's stochastic simulation algorithm (SSA).
This algorithm comes with the advantage of working in continuous time,
i.e. with rates instead of probabilities, which preserves neighboring
state correlations in network simulations and therefore correctly samples
trajectories from the original underlying Markov process.

There's a mountain of literature describing how this algorithm works,
so we will be satisfied with a short description of its underlying
principles in the following.

Homogeneous Poisson Processes
=============================

Gillespie's SSA samples events :math:`e` from an event set :math:`E`.
Each event is associated with a constant rate :math:`\lambda_e`.
The total rate of `any` event happening is given as

.. math::

    \Lambda = \sum_{e\in E} \lambda_e.

Now, suppose we're currently at time :math:`t` and we want to know
a time leap :math:`\tau`. Derived from first principles, this time
leap is distributed according to an exponential distribution

.. math::
    
    \tau \sim \mathcal E (\Lambda).

The event that takes place at this time can be chosen from the event
set with probability

.. math::

    \pi_e = \lambda_e / \Lambda.

After the event takes place, the event set and the event rates
can be updated according to the new situation and time is advanced
as 

.. math::

    t \leftarrow t + \tau.

Inhomogeneous Poisson Processes
===============================

When rates are time-dependent the math becomes a little more
tedious but works out as follows.

Each event is associated with a time-varying rate :math:`\lambda_e(t)`.
The total rate of `any` event happening is given as

.. math::

    \Lambda(t) = \sum_{e\in E} \lambda_e(t).

Now, suppose we're currently at time :math:`t` and we want to know
a time leap :math:`\tau`. We find that the quantity

.. math::

    \Theta(\tau) = \int\limits_t^{t+\tau} dt' \sum_{e\in E}\Lambda(t')

is exponentially distributed as :math:`\Theta \sim \mathcal E(1)`.
Hence, we draw a value for :math:`\tilde\Theta` from the default
exponential distribution and solve the integral :math:`\Theta(\tau)`
for :math:`\tau`.

The event that takes place at this time can be chosen from the event
set with probability

.. math::

    \pi_e(t+\tau) = \lambda_e(t+\tau) / \Lambda(t+\tau).

In `epipack`, two methods exist to solve this integral numerically.
The first method solves the integral :math:`\Theta(\tau)` with
a quadrature method from the scipy.integrate module and applies
a Newton-Raphson root finding method to this numerical integral
function.

The second (and default) method redefines the problem as 
an initial value problem 

.. math::

    \frac{d}{d\tau}\Theta(\tau) = \Lambda(t+\tau), \qquad \Theta(0) = 0

And applies a Runge-Kutta 2(3) method until :math:`\tilde\Theta = \Theta(\tau)`.
This method is faster and has therefore been chosen as the default method.

Both methods yield values of :math:`\tau` that lie within 0.1% of each other.

Temporal Networks
=================

In principle, temporal networks lead to rate functions that change
as step functions. Vestergaard and Génois have used this fact
to come up with a fast simulation algorithm for temporal networks:

"Temporal Gillespie algorithm: Fast simulation of contagion
processes on time-varying networks". C.L. Vestergaard & M. Génois. 
PLoS Computational Biology (2015) 11, e1004579 
(http://arxiv.org/abs/1504.01298).

We do not use this algorithm here though. We simply make use of
the fact that we're dealing with Poisson processes. I.e.
we treat a network as static first and simulate model
processes until we overshoot the time at which the network changes next. 
As soon as we overshoot a time value at which the network changes, we simply
stop at this time value, reset the model with the new network
structure and restart the simulation again.
Doing so does not lead to erroneous results. Since we're
dealing with Poisson processes, we can restart the 
simulation at this time point, because we've already decided
that the next event will happen after this time point.
If the rates would not have changed, 
restarting the simulation is equal to
resampling from the tail of the original distribution
that was cut off after the network change time.

Resetting the entire model/event set every time the network
changes is quite inefficient. Yet, our primary goal here
is to have a framework where we can prototype epidemiological
models in a fast manner while simulation efficacy is left
aside for now.

If you want to simulate canonical models on temporal networks
efficiently, check out tacoma_ where Vestergaard's and Génois's method
is implemented (it's really fast).

.. _`tacoma`: http://tacoma.benmaier.org/




