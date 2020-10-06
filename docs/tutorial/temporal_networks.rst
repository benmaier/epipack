Temporal Networks
=================

`epipack` provides a tiny interface to simulate StochasticEpiModels
on temporal networks, see :mod:`epipack.temporal_networks`.
Note that such simulations are based on the simulation routine
for static networks and therefore numerically correct
but not particularly efficient (the event set is
reset entirely every time the network structure changes).

The idea is once again that prototyping should be fast, i.e.
building new models in a flexible manner, leaving simulation
efficiency aside.

Temporal Network Objects
------------------------

A temporal network is defined by the following properties

- ``N``: the constant number of nodes in the system
- ``t``: A sorted list containing each time point at which 
  the edge set will be updated
- ``tmax``: A final time that marks the end of the experiment
  (not associated with an edge set update).
- ``edges``: An ordered list of edge sets.

    .. code:: python

        edges = [
            [ (0,1), ... ], # edge set for first timestamp
            [ (2,7), ... ], # edge set for second timestamp
            ...         
        ]
- ``directed``: boolean, whether or not edges in edge set
  are supposed to be symmetric
- ``weighted``: boolean, if ``True``, the edges in the edge
  sets per time stamp are expected to be 3-tuples where the 
  last entry is a float value

    .. code:: python

        edges = [
            [ (0,1,1.0), ... ], # edge set for first timestamp
            [ (2,7,0.5), ... ], # edge set for second timestamp
            ...         
        ]

This is a variant of how temporal networks are defined in tacoma_
(in `tacoma`, more temporal network types are possible, but every one
of them only offers unweighted and undirected edges).

In `epipack`, temporal networks are supposed to be constructed using
the :class:`epipack.temporal_networks.TemporalNetwork` class.

Here's an example:

.. code:: python

    pass

Now, you can iterate through the network like so:


.. code:: python

    pass

Note that per default, temporal networks are looped indefinitely. If you don't
want them to loop but to stop the simulation at ``tmax``, pass the keyword
``loop_network=False`` to the constructor

.. code:: python
    
    TemporalNetwork(loop_network=False,*args,**kwargs)

Construct Temporal Networks with Tacoma
---------------------------------------

tacoma_ offers an extensive temporal network analysis frame work,
such that it makes sense to use it to load temporal networks.
For instance:

.. code:: python

    pass

Simulate on StochasticEpiModels
-------------------------------

After loading a temporal network, set up a simulation using
:class:`epipack.temporal_network.TemporalNetworkSimulation`.

It's straight-forward to simulate then:

.. code:: python

    pass

.. _`tacoma`: http://tacoma.benmaier.org/temporal_networks/temporal_network_classes.html
