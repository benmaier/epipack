State-Altering Objects
----------------------

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

Death and birth processes are identified by one of the compartments being ``None``. An involved compartment can only be ``None`` in birth and death processes. Furthermore, link transmission processes have a stricter format: (i) they must be processes where one compartment stays constant while the other compartment changes, and (ii) the compartment remaining constant has to be in the first position on each side of the reaction. For instance,

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

Rates
=====

Linear rates look like this:

.. code:: python

    ( source_compartment, affected_compartment, rate_value ).

quadratic rates look like this:

.. code:: python

    ( coupling_compartment0, coupling_compartment_1, affected_compartment, rate_value ).

The reasoning here is that, sometimes, you just want to create a model by
copying an existing ODE system. Then, it's easier to directly set the rates
instead of converting them to reaction processes in your head.


Events
======
    
    


