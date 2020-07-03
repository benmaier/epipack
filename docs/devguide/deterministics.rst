Deterministic Simulations
-------------------------

Intro
=====

We want to define and integrate a set of :math:`N_c`
ordinary differential equations (ODEs) as

.. math::

    \frac{d}{dt}Y_i = \sum_{j,k} \alpha_{ijk} Y_jY_k + \sum_j \beta_{ij} Y_j + \gamma_i.

To this end, :class:`epipack.deterministic_epi_models.DeterministicEpiModel` contains three
parameter-carrying linear algebra objects, the numpy array ``birth_rates`` that is equal to the vector :math:`\gamma`,
the scipy sparse matrix ``linear_rates`` that is equal to the matrix :math:`\beta` and a list
of scipy sparse matrices ``quadratic_rates`` that contains a scipy sparse matrix in each of 
its entries `i`, which makes it correspond to the tensor :math:`\alpha`.

Choice of Data Structures
=========================

We chose scipy sparse matrices because their API is simple and tailored for
fast execution of dot products. The ODEs are computed using a current-state vector
:math:`X` and dot products between the operators ``linear_rates`` and ``quadratic_rates``.

One might wonder about the specific choice of sparse matrices instead of arrays. The reasoning
here is that these models can be potentially used to set up reaction-diffusion systems where a spatial
dimension is introduced using a (graph) Laplacian. In order to efficiently expand the system
to evolve on every node on a network (discretized point in space), the operators  ``birth_rates``,
``linear_rates``, and ``quadratic_rates`` can be set system-wide using the Kronecker product
(see scipy.sparse.kron).

Setting Rates
=============

There are two ways to set rates

1.  Directly. To this end, use the methods ``model.set_linear_rates`` and ``model.set_quadratic_rates``.
    Rates are encoded as a list of tuples that contain the coupling compartments, the affected compartment,
    and the rate value. Methods whose name begins with ``set`` will override previously set rates.
2.  Through proccesses. To this end, use the methods ``model.set_processes`` or ``model.add_*_processes``.
    Processes are encoded as a list of tuples that contain the coupling compartments, a rate value, 
    and the affected compartment. Methods whose name begins with ``set`` will override previously set rates,
    while methods whose name begins with ``add`` will not override previously set processes.

When rates for quadratic processes are set, the components that are affected by quadratic processes are
saved in `model.affected_by_quadratic_process` as integers. This saves time in situations where there are
many compartments but only few transmission processes.

Integrating ODEs
================

The momenta :math:`dY_i/dt` are automatically computed by means of dot products. They are integrated using
a Runge-Kutta 4(5) method with adaptive step size (Dormand-Prince) (see :func:`epipack.integrators.integrate_dopri5`).


