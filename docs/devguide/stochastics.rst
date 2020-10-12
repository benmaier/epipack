.. _dev-stochastic-sims:

Stochastic Simulations
----------------------

The whole generalized algorithm is based on this phenomenal paper by
St-Onge `et al.`:

"Efficient sampling of spreading processes on complex networks using a composition and rejection algorithm", G.St-Onge, J.-G. Young, L. Hébert-Dufresne, and L. J. Dubé, Comput. Phys. Commun. 240, 30-37 (2019), http://arxiv.org/abs/1808.05859.

In fact, the simulations are tremendously sped up if you install
their samplable set that `epipack` will use automatically
if it succeeds at importing it (by default,
if ``SamplableSet`` is not installed, `epipack` will fall back on 
a numpy-based internal ``MockSamplableSet`` that mimicks ``SamplableSet``'s
behavior).

Node-Based Events
=================

For StochasticEpiModels, processes are converted to node-based events.
Here, the algorithm needs to know which events a node can take part in
leading the active (transmitting) role or a transitioning role.

In this particular simulation framework, events are entirely node-based,
in the sense that events can only happen based on compartments
that nodes carry. This is not an approximation: link-based 
infection events are simply grouped together and associated with the infecting nodes.

Hence, node-based event tuples are only connected to single compartments.
Two class attributes

- ``link_transmission_events``, and
- ``node_transition_events``

carry event tuples that classify transition descriptors. Each of these
lists has :math:`N_c` entries (once for each compartment) and each
entry has two elements. The first element is a :math:`N_\mathrm{events,C} \times 3`
matrix where each row contains all three changes associated with one of
:math:`N_\mathrm{events,C}` events.
The second element is a single-row array that contains a rate value for each
of the elements encoded in the first matrix.

This must be very confusing to read. Here is an example. Let's say
we have diseases `A` and `B` and one type of susceptible compartment `S` such that

.. math::

    A + S &\stackrel{\eta}{\longrightarrow} A + A\\
    B + S &\stackrel{\eta}{\longrightarrow} B + B.

Also, `A` and `B` recover with different rates

.. math::

    A &\stackrel{\rho_A}{\longrightarrow} S\\
    B &\stackrel{\rho_B}{\longrightarrow} S.

Suppose we've set up a model like

.. code:: python

    model = StochasticEpiModel(['S', 'A', 'B'],N=1000)

Let's define the link transmission events first

.. code:: python

    model.set_link_transmission_processes([
        ("A", "S", eta, "A", "A"),
        ("B", "S", eta, "B", "B"),
    ])

Now, we assume that ``eta = 1.0``.
Consequently, ``model.link_transmission_events`` looks like this:

.. code:: python

    model.link_transmission_events = [
        (),
        (
            array([ [1, 0, 1] ]),
            array([ 1.0 ]),
        ),
        (
            array([ [2, 0, 2] ]),
            array([ 1.0 ]),
        ),
    ]

The first entry of this list is an empty tuple. This is because
susceptible nodes cannot infect anybody.

The second entry (second compartment is `A`) of this list is a 2-tuple. 
Its first element
contains a matrix with a single row and three columns. The single
row represents the single infection event a node of compartment `A`
can cause. ``[1,0,1]`` represents that the infection event: a node
of compartment 1 (represents `A`) coupled with a node of compartment 0
(represents `S`) lets `S` transition to `A` (compartment 0 to compartment 1,
respectively). The array ``array([ 1.0 ])`` contains the single rate 
with which this single event can take place.

Similarly, the third entry (associated with nodes of compartment `B`),
contains the `B` + `S` event and concurrent infection rate.

Now, let's say :math:`\rho_A=1/2` and :math:`\rho_B=1/4` and we set 
the transition events:

.. code:: python

    model.set_node_transition_processes([
        ("A", rho_A, "S"),
        ("B", rho_B, "S"),
    ])

And we find ``model.node_transition_events`` to take the following shape:

.. code:: python

    [
        (),
        (
            array([ [-1, 1, 0] ]),
            array([ 0.5 ]),
        ),
        (
            array([ [-1, 2, 0] ]),
            array([ 0.25 ]),
        ),
    ]

Again, susceptibles do not transition spontaneously. I.e. the first
entry of this list is an empty tuple.

The second entry contains (a) a matrix that describes a single event
(one row). This event is ``[-1, 1, 0]``. The first ``-1`` represents
a non-existing infection compartment: the compartment 1 (represents
`A`) transitions spontaneously to compartment 0 (represents `S`).
Also, this second entry contains (b) an array with a single element:
the recovery rate associated with this single transition event.

The third entry codifies the ``B -> S`` event in a similar manner.

The definitions of conditional transmission events work in a similar
way. Instead of rates, the arrays on the second positions contain probabilities.

Compartment-Based Events
========================

After both node and link processes have been defined, they are zipped together
to build ``model.node_and_link_events`` (in the internal method ``model._zip_events()``).

This attribute looks similar to ``model.node_transition_events`` and
``model.link_transmission_events`` but event matrices are stacked 
and event rates are concatenated. Also, each compartment-tuple contains 
an additional entry where the range of all link events is encoded by means
of two indices.

In our example, ``model.node_and_link_events`` looks like

.. code:: python

    [
        (),
        (
            array([ [-1, 1, 0],
                    [ 1, 0, 1] ]),
            array([ 0.5, 1.0 ]),
            [ 1, 2 ],
        ),
        (
            array([ [-1, 2, 0],
                    [ 2, 0, 2] ]),
            array([ 0.25, 1.0 ]),
            [ 1, 2 ],
        ),
    ]

The algorithm saves the indices in order to scale the rate of these events
with a node's out-degree.

Every time a node changes its compartment, the corresponding event set of
this compartment is loaded from ``model.node_and_link_events``, and 
the rate vector's entries in the range of the specified link event range
will be scaled by the node's out-degree. The sum of this vector is then
passed to the global event set. The vector itself is saved 
``model.node_event_probabilities``. After the global event set has
been sampled for an event and a node has been chosen, 
a specific node-event is sampled. If this event is a node event, 
it simply happens. If it is a link event, a random neighbor is sampled
proportional to its link weight
If the neighbor has the right compartment, the infection event takes place
and time is advanced. If the neighbor does not have the right compartment,
the proposed event is rejected and time is advanced nevertheless.

One may wonder whether such a procedure truly reflects the spirit
of the rejection algorithm. In the following we present an
example that shows that this is indeed the case.

Let's discuss a test case where a single node of compartment `A` and index 0 can infect `S`-nodes with rate ``aS = 2.0`` and `B` with rate ``aB = 0.5``. Links are set up like

.. code:: python

    [
      (0, 1, 10.0)
      (0, 2, 1.0)
      (0, 3, 1.0)
    ]

with nodal compartments

.. code:: python

    {
     0: 'A',
     1: 'S',
     2: 'B',
     3: 'B',
    }

Now these are the true events that may happen:

.. code:: python

    [
      ( 1, '->', 'A', 20.0),
      ( 2, '->', 'A', 0.5),
      ( 3, '->', 'A', 0.5),
    ]

with total event rate 21.0.

However, these are the events epipack's algorithm assumes might happen 
(as per the rejection sampling algorithm):

.. code:: python

    [
      (1, '->', 'A', 20.0),
      (2, '->', 'A', 2.0),
      (3, '->', 'A', 2.0),
      (1, '->', 'B', 5.0),
      (2, '->', 'B', 0.5),
      (3, '->', 'B', 0.5),
    ]

with total rate 30.0.

In principle, the algorithm has to choose one of the events from this list and then reject it if can't happen (i.e. if the neighboring node of the chosen event does not have the correct compartment). Instead, what it does is to sample
(i) a general event, i.e. either 'A' with rate 24.0 or 'B' with rate 6.0. Then, it samples (ii) a neighbor according to the link's weight that connects the origin node to this neighbor. If the neighbor has a compartment that fits with the previously sampled event, the event can take place. If not, the event is rejected, time is advanced, and a new event is sampled.
This second method can be interpreted as deciding first from which bulk of this event super set we sample from and deciding for an event from this bulk afterwards.

Hence, it doesn't matter whether a single event is sampled from the entire list or whether it's decided first which bulk of this list the event will be chosen from. After the decision of the bulk has been made (i.e. by choosing the target compartment), only the link weight is important in determining which event is chosen.
