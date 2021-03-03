"""
Classes to define temporal networks and run stochastic simulations
on them.
"""

import numpy as np

class TemporalNetwork():
    """
    A simple temporal network class.

    Parameters
    ==========
    N : int
        Static number of nodes in the network.
    t : list
        An increasingly ordered list of times.
    tmax : float
        The time at which the recording of network changes
        has concluded (the final time of the
        temporal network).
    edge_lists : list
        A list of lists of tuples. Each entry i
        of this list represents an edge list.
        Each entry j of an edge list is a tuple
        of format

        .. code:: python

            ( source, target, weight )

        if ``weighted=True``, or

        .. code:: python

            ( source, target )

        otherwise. ``source`` and ``target``
        are supposed to be of unsigned integer type
        in range ``[0,N)``.
    weighted : bool, default = False
        Whether or not ``edge_lists`` contain edge
        tuples with weights
    directed : bool, default = False
        Whether or not edge tuples will be considered
        to be directed.
    loop_network : bool, default = True
        If `True`, the temporal network will be looped
        indefinitely when iterated.

    Example
    =======

    >>> edges = [ [ (0,1) ], [ (0,1), (0,2) ], [] ]
    >>> temporal_network = TemporalNetwork(3,[0,0.5,0.6],1.0,edges)
    >>> for edge_list, t, next_t in temporal_network:
    ...     if t >= 3:
    ...         break
    ...     print(t, next_t, edge_list)
    ...
    0 0.5 [(0, 1, 1.0)]
    0.5 0.6 [(0, 1, 1.0), (0, 2, 1.0)]
    0.6 1.0 []
    1.0 1.5 [(0, 1, 1.0)]
    1.5 1.6 [(0, 1, 1.0), (0, 2, 1.0)]
    1.6 2.0 []
    2.0 2.5 [(0, 1, 1.0)]
    2.5 2.6 [(0, 1, 1.0), (0, 2, 1.0)]
    2.6 3.0 []
    

    Attributes
    ==========
    N : int
        Total and constant number of nodes in the system.
    t : list
        An increasingly ordered list of times.
    tmax : float
        The time at which recording of edge changes
        has concluded (the final time of the
        temporal network).
    edge_lists : list
        A list of lists of tuples. Each entry i
        of this list represents an edge list.
        Each entry j of an edge list is a tuple
        of format

        .. code:: python

            ( source, target, weight )

        if ``weighted=True``, or

        .. code:: python

            ( source, target )

        otherwise. ``source`` and ``target``
        are supposed to be of unsigned integer type
        in range ``[0,N)``.
    weighted : bool
        Whether or not ``edge_lists`` contain edge
        tuples with weights
    directed : bool
        Whether or not edge uples will be considered
        to be directed.
    loop_network : bool
        If `True`, the temporal network will be looped
        indefinitely when iterated.
    """

    def __init__(self,
                 N,
                 edge_lists,
                 t,
                 tmax,
                 weighted=False,
                 directed=False,
                 loop_network=True,
                 ):

        assert([ a == b for a, b in zip(sorted(t),t) ])
        assert(t[-1] < tmax)
        assert(len(edge_lists) == len(t))

        _t = np.array(t)
        assert(np.all(_t[1:] - _t[:-1] > 0))

        self.t = t
        self.tmax = tmax
        self.N = N
        self.edge_lists = edge_lists
        self.weighted = weighted
        self.directed = directed
        self.T = tmax - t[0]
        self.loop_network = loop_network

    @classmethod
    def from_tacoma(cls, edge_lists, loop_network=True):
        """Initiate from a tacoma.edge_lists instance"""
        return cls(edge_lists.N,
                   edge_lists.edges,
                   edge_lists.t,
                   edge_lists.tmax,
                   weighted=False,
                   directed=False,
                   loop_network=loop_network,
                   )

    def t0(self):
        """Obtain the initial time"""
        return self.t[0]

    def mean_out_degree(self):
        """Obtain the temporally averaged mean out-degree"""
        if not self.weighted:
            k = 0.0
            if not self.directed:
                factor = 2
            else:
                factor = 1
            for edges, t, next_t in self:
                if self._has_looped():
                    break
                k += (next_t - t) * factor*len(edges)/self.N
            k /= self.T
        else:
            k = np.zeros(self.N,dtype=float)
            for edges, t, next_t in self:
                if self._has_looped():
                    break
                dt = next_t - t
                for u, v, w in edges:
                    k[u] += w * dt
                    if not self.directed:
                        k[v] += w * dt
            k = k.mean() / self.T
        return k

    def _has_looped(self):
        """Whether or not the network has looped at least once"""
        return self._Delta_t >= self.T

    def __iter__(self):
        self._ndx = 0
        self._Delta_t = 0
        return self

    def __next__(self):

        if self._ndx < len(self.t):

            t0 = self.t0()

            # get the current time (including potential loops)
            t = self.t[self._ndx] - t0
            t += self._Delta_t + t0

            # get the current edges and add default weight = 1.0 if unweighted
            edges = self.edge_lists[self._ndx]
            if not self.weighted:
                edges = [ (u,v,1.0) for u, v in edges ]

            # increase the iteration pointer and calculate the next
            # time when changes happen
            self._ndx += 1
            if self._ndx < len(self.t):
                next_t = self.t[self._ndx] - t0
            else:
                next_t = self.tmax - t0
            next_t += self._Delta_t + t0

            # if the end of the list is reached and the
            # network is supposed to be looped,
            # reset the iteration pointer and
            # and increase the time loop delta.
            if self._ndx == len(self.t) and self.loop_network:
                self._ndx = 0
                self._Delta_t += self.T

        # if the end of the list is reached and
        # the network is not supposed to be looped,
        # stop the iteration
        else:
            raise StopIteration

        return edges, t, next_t

class TemporalNetworkSimulation():
    """
    Simulation of a StochasticEpiModel on a
    temporal network.

    Parameters
    ==========
    temporal_network : TemporalNetwork
        A temporal network object.
    stochastic_epi_model : epipack.stochastic_epi_models.StochasticEpiModel
        An instance of StochasticEpiModel, initialized
        with processes and initial conditions.

    Example
    =======

    >>> model = StochasticEpiModel(["S","I","R"],3)\\
    >>>             .set_link_transmission_processes([
    >>>                     ("I", "S", 1.0, "I", "I"),
    >>>                 ])\\
    >>>             .set_node_transition_processes([
    >>>                     ("I", 2.0, "R"),
    >>>                 ])\\
    >>>             .set_node_statuses([1,0,0])
    >>> temporal_network = TemporalNetwork(3,edges,[0,0.5,1.2],1.5)
    >>> sim = TemporalNetworkSimulation(temporal_network, model)
    >>> sim.simulate(tmax=100)
    [ 0.          0.38740794  0.8210494   3.60987397  9.46213129 12.14301704] {'S': array([2., 1., 0., 0., 0., 0.]), 'I': array([1., 2., 3., 2., 1., 0.]), 'R': array([0., 0., 0., 1., 2., 3.])}

    Attributes
    ==========
    temporal_network : TemporalNetwork
        A temporal network object.
    model : epipack.stochastic_epi_models.StochasticEpiModel
        An instance of StochasticEpiModel, initialized
        with processes and initial conditions.
    """

    def __init__(self,temporal_network,stochastic_epi_model):
        self.temporal_network = temporal_network
        self.model = stochastic_epi_model
        self._initial_node_status = np.array(self.model.node_status)

        assert(self.temporal_network.N == self.model.N_nodes)

    def simulate(self,
                 tmax,
                 return_compartments=None,
                 max_unsuccessful=None,
                 sample_only_on_network_updates=False,
                 ):
        """
        Simulate a StochasticEpiModel on a temporal network.

        Parameters
        ----------
        tmax : float
            maximum length of the simulation
        return_compartments : list of compartments, default = None:
            The compartments for which to return time series.
            If ``None``, all compartments will be returned.
        max_unsuccessful : int, default = None
            The number of unsuccessful events after which the
            true total event rate will be evaluated (it might happen
            that a network becomes effectively disconnected while
            nodes are still associated with a maximum event rate).
            If ``None``, this number will be set equal to the number of nodes.
        sample_only_on_network_updates : funtion, default = None
            A function that's called when a sample is taken

        Returns
        -------
        t : numpy.ndarray
            times at which compartment counts have been sampled
        result : dict
            Dictionary mapping a compartment to a time series of its count.
        """

        if return_compartments is None:
            return_compartments = self.model.compartments

        time = [ self.temporal_network.t0() ]
        result = { C: [ 
                np.count_nonzero(
                    self.model.node_status==self.model.get_compartment_id(C)
                    )
            ] for C in return_compartments }

        for edges, t, next_t in self.temporal_network:

            if t >= tmax or self.model.simulation_has_ended():
                break

            self.model.set_network(
                    N_nodes=self.temporal_network.N,
                    edge_weight_tuples=edges,
                    directed=self.temporal_network.directed,
                    )

            _t, _res = self.model.simulate(
                    t0=t,
                    tmax=next_t,
                    return_compartments=return_compartments,
                    max_unsuccessful=max_unsuccessful,
                    stop_simulation_on_empty_network=False,
                    )

            if next_t > tmax:
                next_t == tmax

            # only save result if in fact anything changed (i.e.
            # if the number of sampling times exceeds one.
            if len(_t) > 1:
                if sample_only_on_network_updates:
                    time.append(next_t)
                    for C, timeseries in _res.items():
                        result[C].append(timeseries[-1])
                else:
                    # save all sampling events besides the first
                    time.extend(_t.tolist()[1:])
                    for C, timeseries in _res.items():
                        result[C].extend(timeseries.tolist()[1:])

        time = np.array(time)
        for C in result:
            result[C] = np.array(result[C])

        return time, result

    def reset(self,node_status=None):
        """Reset the model"""
        if node_status is not None:
            self._initial_node_status = np.array(node_status)

        self.model.set_node_statuses(self._initial_node_status)



                    
if __name__=="__main__": # pragma: no cover
    edges = [ [ (0,1) ], [ (0,1), (0,2) ], [] ]
    temporal_network = TemporalNetwork(3,edges,[0,0.5,0.6],1.0)
    for edge_list, t, next_t in temporal_network:
        if t >= 3.0:
            break
        print(t, next_t, edge_list)

    print("Delta T =",temporal_network._Delta_t)
    for edge_list, t, next_t in temporal_network:
        print("DeltaT =",temporal_network._Delta_t)
        break


    from epipack import StochasticEpiModel

    infection_rate = 1.0
    recovery_rate = 0.2
    model = StochasticEpiModel(["S","I","R"],3)\
                .set_link_transmission_processes([
                        ("I", "S", infection_rate, "I", "I"),
                    ])\
                .set_node_transition_processes([
                        ("I", recovery_rate, "R"),
                    ])\
                .set_node_statuses([1,0,0])

    temporal_network = TemporalNetwork(3,edges,[0,0.5,1.2],1.5)
    sim = TemporalNetworkSimulation(temporal_network, model)
    t, res = sim.simulate(100)
    print(t, res)
    sim.reset()
    print(sim.model.node_status)
    t, res = sim.simulate(100)
    print(t, res)

    import matplotlib.pyplot as pl
    from tqdm import tqdm

    N_meas = 10000
    #model = StochasticEpiModel(["S","I","R"],3)\
    #            .set_link_transmission_processes([
    #                    ("I", "S", infection_rate, "I", "I"),
    #                ])\
    #            .set_node_transition_processes([
    #                    ("I", recovery_rate, "R"),
    #                ])\
    model.set_node_statuses([1,0,0])

    #temporal_network = TemporalNetwork(3,edges,[0,0.5,1.5],3.0)
    #sim = TemporalNetworkSimulation(temporal_network, model)

    taus = []
    for meas in tqdm(range(N_meas)):
        sim.reset()
        t, res = sim.simulate(1000)
        if t[-1] == 0:
            continue
        else:
            taus.append(t[1])

    pl.hist(taus,bins=200,density=True)

    def rate(t):
        t = t % 1.5
        if t < 0.5:
            return infection_rate + recovery_rate
        elif t < 1.2:
            return 2*infection_rate + recovery_rate
        elif t < 1.5:
            return recovery_rate

    from scipy.integrate import cumtrapz

    t = np.linspace(0,max(taus),10000)
    rates = np.array([rate(_t) for _t in t])
    RATES = cumtrapz(rates,t,initial=0.0)
    pl.plot(t,rates*np.exp(-RATES))
    #pl.yscale('log')
    pl.show()
