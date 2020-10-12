State-Altering Tuples
---------------------

Processes
=========

Throughout the project, `processes` will refer to tuples that mimick reaction equations.
The tuples contain references to compartments involved in a reaction and a rate (or a
placeholder for a rate).

There are five allowed processes

.. code:: python

    # transition process
    ( source_compartment, rate, target_compartment),

    # transmission process
    ( coupling_compartment_0, coupling_compartment_1, rate, target_compartment_0, target_ccompartment_1),
    # fission process
    ( source_compartment, rate, target_compartment_0, target_ccompartment_1),
    
    # fusion process
    ( source_compartment_0, source_compartment_1, rate, target_compartment),

    # death process
    ( source_compartment, rate, None),

    # birth process
    ( None, rate, target_compartment),

``epipack`` identifies the process based on the length of the tuple, the rate (an entry not being a compartment) and. For ``SymbolicEpiModel``, rates can be equal to compartments, but then processes have to be set with ``ignore_rate_position_checks``. This might change in the future to be default behavior for ``SymbolicEpiModel``.

Death and birth processes are identified by one of the compartments being ``None``. An involved compartment can only be ``None`` in birth and death processes. 

For a StochasticEpiModel, link transmission processes have a stricter format: (i) they must be processes where one compartment stays constant while the other compartment changes, and (ii) the compartment remaining constant has to be in the first position on each side of the reaction. For instance,

.. code:: python

    ( "I", "S", rate, "I", "I" )

is a valid transition, but ``("S", "I", rate, "I", "I")`` is not. The reasoning behind this restriction is to make sure that the user really thinks about which node is reacting and transmitting.

For ``StochasticEpiModel``, only node transition and link transmission processes are allowed. Additionally, there is the possibility to set conditional link transmission processes that are executed immediately after a base process happened. Base reactions are encoded like this

.. code:: python

    (source_base, "->", target_base) # or
    (transmitting, source, "->", transmitting, target)

Conditional transmission process are always formulated as dictionaries with lists of processes.

.. code:: python

    {
        (source_base, "->", target_base) : [
            (target_base, source, "->", target_base, target)
        ]
    }


This means that after a node transitions from ``source_base`` to ``target_base``, its neighbors will be checked and **every** neighbor
that is of compartment ``source`` will be transitioned to compartment ``target``. Note the constancy of ``target_base`` in both
the base process and the conditional process. This a necessary condition for conditional transmission processes because they
can only happen for nodes that change their compartments. We might relax this condition in the future, should it be necessary
for a transmitting node to be involved in a conditional process, too.

We can relax the **every neighbor** condition by introducing probabilities. For instance, in

.. code:: python

    {
        (source_base, "->", target_base) : [
            (target_base, source, p, target_base, target) # case A
        ]
    }

the conditional transition happens for every ``source``-neighbor with probability `p`. Otherwise, the ``source``-neighbor will
remain a ``source``-neighbor.

One may introduce several events for ``source``-neighbors, each with certain probabilities, like

.. code:: python

    {
        (source_base, "->", target_base) : [
            (target_base, source, p, target_base, target) # case A
            (target_base, source, q, target_base, target) # case B
        ]
    }

In this case, ``epipack`` adds a "nothing happens"-process ``(target_base, source, 1-p-q, target_base, target)`` automatically such 
that any of the possible processes happens to the ``source``-neighbor (with corresponding probability :math:`p`, :math:`q`, or
:math:`1-p-q`).

Events
======

Event tuples are used in the default implementations of
EpiModels (EpiModel, StochasticEpiModel), because they're
flexible enough that we can construct both mean-field
ODEs as well as stochastic simulations. Events are defined
in a way such that coupling of one or two compartments leads
to a change in the overall state by applying a difference
vector to the current state as

.. math::

    \Delta Y^{(e)} = ( +1, 0, 0, -1, ... ).

Hence, for event tuples we need to define

1. Coupling compartments
2. A rate value
3. The state change vector. 

We do that as follows:

.. code:: python

    events = [
        (
            (coupling_compartment_0, coupling_compartment_1,),
            rate_value,
            ( (affected_compartment_0, 1), (affected_compartment_1, -1), ... )
        )
    ]

For linear events, the first entry of an event tuple
will just be a single-element-tuple. All un-mentioned
compartments in the event tuple's last entry will be
assumed to not change (a zero entry in the state change
vector).

For instance, for an SEIR model, we would set

.. code:: python

    [
        (
            ('E',),
            1/incubation_time,
            ( ('E', -1), ('I', +1) )
        ),
        (
            ('I',),
            1/infectious_period,
            ( ('I', -1), ('R', +1) )
        )
    ]

For infection events, e.g. in an SEIR model, we would set
    
.. code:: python

    [
        (
            ('S','I'),
            infection_rate,
            ( ('S', -1), ('E', +1) )
        ),
    ]

Rates
=====

Rate tuples are used for constant-rate EpiModels like
MatrixEpiModel and SymbolicMatrixEpiModel. Only
constant values can be set in MatrixEpiModel, because
it makes use of scipy's sparse matrix API which
is quite efficient for large systems.

Linear rates look like this:

.. code:: python

    ( source_compartment, affected_compartment, rate_value ).

For instance, for an SEIR model, we would set

.. code:: python
    
    [
        ('E', 'E', -1/incubation_time),
        ('E', 'I', +1/incubation_time),
        ('I', 'I', -1/infectious_period),
        ('I', 'R', +1/infectious_period),
    ]

Quadratic rates look like this:

.. code:: python

    ( coupling_compartment0, coupling_compartment_1, affected_compartment, rate_value ).

E.g. for a model where both asymptomatic infecteds `A` as well as
symptomatic infecteds `I` could infect susceptibles, we would define

.. code:: python

    [
        ('I', 'S', 'S', -inf_to_inf_rate),
        ('I', 'S', 'I', +inf_to_inf_rate),
        ('I', 'S', 'S', -inf_to_asymp_rate),
        ('I', 'S', 'A', +inf_to_asymp_rate),
        ('A', 'S', 'S', -asymp_to_asymp_rate),
        ('A', 'S', 'A', +asymp_to_asymp_rate),
        ('A', 'S', 'S', -asymp_to_inf_rate),
        ('A', 'S', 'I', +asymp_to_inf_rate),
    ]

The reasoning here is that, sometimes, you just want to create a model by
copying an existing ODE system. Then, it's easier to directly set the rates
instead of converting them to reaction processes in your head.

Node-Based Events
=================

For StochasticEpiModels, processes are converted to node-based events.
Here, the algorithm needs to know which events a node can take part in
leading the active (transmitting) role or a transitioning role.
