"""
Contains a general base class to define 
stochastic epidemiological models.
"""
import numpy as np


# Try to import the original SamplableSet,
# but if that doesn't work, use the mock version
# that's implemented in this package
try:
    from SamplableSet import SamplableSet
except ImportError as e:
    from MockSamplableSet import MockSamplableSet as SamplableSet
    raise ImportWarning("Couldn't find the efficient implementation of `SamplableSet` (see github.com/gstonge/SamplableSet). Proceeding with less efficient implementation.")

#from MockSamplableSet import MockSamplableSet as SamplableSet

# define some integer pointers to positions in lists/tuples

# positions of compartments in event tuples
_TRANSMITTING_COMPARTMENT = 0
_AFFECTED_SOURCE_COMPARTMENT = 1
_AFFECTED_TARGET_COMPARTMENT = 2

# position of the edge weight in weighted edge tuples
_EDGE_WEIGHT = 2

# positions of items in the event lists
_EVENTS = 0
_RATES = 1
_LINK_PROCESS_INDICES = 2

class StochasticEpiModel():
    """
    A general class to define any 
    compartmental epidemiological model
    that can be run in a well-mixed
    system or on a weighted, directed network.
    By default, the epidemiological process is considered
    to run in a well-mixed system.

    Parameters
    ----------
    compartments : :obj:`list` of :obj:`string`
        A list containing compartment strings.
    N_nodes : int
        The number of nodes in the system.
    edge_weight_tuples : list of tuples of (int, int, float), default = None
        Choose this ption
        The links along which transmissions can take place.

            [ (source_id, target_id, weight), ... ]

    directed : bool, default = False
        If `directed` is False, each entry in the edge_weight_tuples
        is considered to equally point from `target_id` to `source_id`
        with weight `weight`.
    well_mixed_mean_contact_number : int, default = 1
        By default, the epidemiological process is considered to run
        in a well-mixed population where for each link-transmission
        event, a node is assumed to have contact to exactly one
        other node.


    Attributes
    ----------

    compartments : :obj:`list` of :obj:`string`
        A list containing strings that describe each compartment,
        (e.g. "S", "I", etc.).
    N_comp : :obj:`int`
        Number of compartments (including population number)
    node_status : numpy.ndarray of int
        Each entry gives the compartment that the
        corresponding node is part of.


    Example
    -------

    .. code:: python
        
        >>> epi = StochasticEpiModel(["S","I","R"],10)
        >>> print(epi.compartments)
        [ "S", "I", "R" ]


    """

    def __init__(self,compartments,N,edge_weight_tuples=None,directed=False,well_mixed_mean_contact_number=1):
        """
        """


        # initial conditions
        self.y0 = None

        # compartment definitions
        self.compartments = list(compartments)
        self.N_comp = len(self.compartments)

        # lists defining events and their rates
        self.node_transition_events = None
        self.link_transmission_events = None
        self.node_and_link_events = None
        self.conditional_link_transmission_events = None

        # check whether to initiate as a network or well-mixed simulation
        if edge_weight_tuples is not None:
            self.set_network(N, edge_weight_tuples, directed)
        else:
            self.set_well_mixed(N,well_mixed_mean_contact_number)

    def set_network(self,N_nodes,edge_weight_tuples,directed=False):
        """
        Define the model to run on a network.

        Parameters
        ---------
        N_nodes : int
            Number of nodes in the system
        edge_weight_tuples : list of tuple of (int, int, float)
            The links along which transmissions can take place.

                [ (source_id, target_id, weight), ... ]

        directed : bool, default = False
            If `directed` is False, each entry in the edge_weight_tuples
            is considered to equally point from `target_id` to `source_id`
            with weight `weight`.

        """

        self.N_nodes = int(N_nodes)

        # min weights and max weights have to be set for the efficient
        # sampling within sets
        min_weight = min([ e[_EDGE_WEIGHT] for e in edge_weight_tuples ])
        max_weight = max([ e[_EDGE_WEIGHT] for e in edge_weight_tuples ])

        # construct network as list of neighbor dictionaries
        graph = [ {} for n in range(N_nodes) ]
        for source, target, weight in edge_weight_tuples:
            graph[source][target] = weight
            if not directed:
                graph[target][source] = weight

        # construct  network as list of SamplableSets, because we want to sample neighbors
        self.graph = [ SamplableSet(min_weight,max_weight,neighbors) for neighbors in graph ]

        # this model is a network model
        self.is_network_model = True

        # compute out degrees in order to compute the maximum reaction rate for link processes
        self.out_degree = np.array([ neighbors.total_weight() for neighbors in self.graph ], dtype=int)

        return self

    def set_well_mixed(self,N_nodes,mean_contact_number):
        """
        Define the model to run in a well-mixed system. 
        
        Parameters
        ---------
        N_nodes : int
            Number of nodes in the system
        mean_contact_number : int
            Each node is assumed to be in contact with `mean_contact_number`
            other nodes at all times. These neighbors will be sampled randomly
            from the set of all remaining nodes every time a link transmission
            event happens.
        """

        if mean_contact_number != int(mean_contact_number):
            raise TypeError("`mean_contact_number` must be of type `int`")

        # define homogeneous degree across the network
        self.N_nodes = int(N_nodes)
        self.k0 = int(mean_contact_number)

        # compute out degrees in order to compute the maximum reaction rate for link processes
        self.out_degree = np.ones((self.N_nodes,),dtype=int) * self.k0

        self.is_network_model = False

        return self

    def get_compartment_id(self,C):
        """Get the integer ID of a compartment ``C``"""
        return self.compartments.index(C)

    def get_compartment(self,iC):
        """Get the compartment, given an integer ID ``iC``"""
        return self.compartments[iC]

    def set_node_transition_processes(self,process_list):
        """
        Define the linear node transition processes between compartments.

        Parameters
        ==========

        process_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions rates in the following format:

            .. code:: python

                [
                    ("source_compartment", rate, "target_compartment" ),
                    ...
                ]

        Example
        -------

        For an SEIR model.

        .. code:: python

            epi.set_node_transition_processes([
                ("E", symptomatic_rate, "I" ),
                ("I", recovery_rate, "R" ),
            ])

        """

        # each compartment is associated with a list of events that it can be
        # involved in (event = first entry)
        # Each event is associated with a rate (rate = second entry)
        self.node_transition_events = [ [[],[]] for n in range(self.N_comp) ]

        # iterate through processes
        for source, rate, target in process_list:

            _s = self.get_compartment_id(source)
            _t = self.get_compartment_id(target)

            # add event and rate to the list of events for this compartment
            self.node_transition_events[_s][_EVENTS].append((-1,_s,_t))
            self.node_transition_events[_s][_RATES].append(rate)

        # convert to numpy arrays
        for _c0, events in enumerate(self.node_transition_events):
            self.node_transition_events[_c0][_EVENTS] = np.array(events[_EVENTS])
            self.node_transition_events[_c0][_RATES] = np.array(events[_RATES])

        # if link events have been set before, sew both events together
        # to form a unified event list for each compartment
        if self.link_transmission_events is not None:
            self._zip_events()

        return self

    def set_link_transmission_processes(self,process_list):
        r"""
        Define link transmission processes between compartments.

        Parameters
        ----------
        process_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions rates in the following format:

            .. code:: python

                [
                    ("source_compartment", 
                     "target_compartment_initial",
                     rate 
                     "source_compartment", 
                     "target_compartment_final", 
                     ),
                    ...
                ]

        Example
        -------

        For an SEIR model.

        .. code:: python

            epi.set_link_transmission_processes([
                ("I", "S", +1, "I", "E" ),
            ])

        """

        # each compartment is associated with a list of events that it can be
        # involved in (event = first entry)
        # Each event is associated with a rate (rate = second entry)
        self.link_transmission_events = [ [[],[]] for n in range(self.N_comp) ]

        # iterate through processes
        for coupling0, coupling1, rate, affected0, affected1 in process_list:

            if coupling0 != affected0:
                raise ValueError("In process",
                                  coupling0, coupling1, "->", affected0, affected1,
                                  "The source (infecting) compartment", coupling0, "and first affected compartment (here: ", affected0,
                                  "), must be equal on both sides of the reaction equation but are not.",
                                  )

            _c0 = self.get_compartment_id(coupling0)
            _s = self.get_compartment_id(coupling1)
            _t = self.get_compartment_id(affected1)

            # add event and rate to the list of events for this compartment
            self.link_transmission_events[_c0][_EVENTS].append((_c0,_s,_t))            
            self.link_transmission_events[_c0][_RATES].append(rate)

        # convert to numpy arrays
        for _c0, events in enumerate(self.link_transmission_events):
            self.link_transmission_events[_c0][_EVENTS] = np.array(events[_EVENTS])
            self.link_transmission_events[_c0][_RATES] = np.array(events[_RATES])

        # if node events have been set before, sew both events together
        # to form a unified event list for each compartment
        if self.node_transition_events is not None:
            self._zip_events()

        return self

    def set_conditional_link_transmission_processes(self,process_dict):
        r"""
        Define link transmission processes between compartments.

        Parameters
        ----------
        process_dict : :obj:`list` of :obj:`tuple`
            A dictionary of tuples that contains conditional transmission events in the following format:

            .. code:: python

                {  
                    conditional_reaction: [
                        ("source_compartment", 
                         "target_compartment_initial",
                         "source_compartment", 
                         "target_compartment_final", 
                         ),
                    ...
                   ]
                }

        Example
        -------

        For an SEIR model.

        .. code:: python

            epi.set_link_transmission_processes([
                ("I", "S", +1, "I", "E" ),
            ])

        """

        # each compartment is associated with a list of events that it can be
        # involved in (event = first entry)
        # Each event is associated with a rate (rate = second entry)
        self.conditional_link_transmission_events = {}

        # iterate through processes
        for triggering_event, triggered_events in process_dict.items():

            if len(triggering_event) == 5:
                coupling0, coupling1, _, affected0, affected1 = triggering_event
                if coupling0 != affected0:
                    raise ValueError("In process",
                                      coupling0, coupling1, "->", affected0, affected1,
                                      "The source (infecting) compartment", coupling0, "and first affected compartment (here: ", affected0,
                                      "), must be equal on both sides of the reaction equation but are not.",
                                      )
                _c0 = self.get_compartment_id(coupling0)
                _s = self.get_compartment_id(coupling1)
                _t = self.get_compartment_id(affected1)
            elif len(triggering_event) == 3:
                source, _, target = triggering_event
                _c0 = -1
                _s = self.get_compartment_id(source)
                _t = self.get_compartment_id(target)
            else:
                raise ValueError("Only node transition or link transmission events are allowed to trigger conditional link transmission events (invalid event:", triggering_event,'"')

            event = (_c0, _s, _t)
            self.conditional_link_transmission_events[event] = {}


            for triggered_event in triggered_events:
                # link event
                if len(triggered_event) != 5:
                    raise ValueError("Only link transmission events are allowed to trigger conditional link transmission events (invalid event:", triggered_event)

                coupling0, coupling1, _, affected0, affected1 = triggered_event

                if coupling0 != affected0:
                    raise ValueError("In process",
                                      coupling0, coupling1, "->", affected0, affected1,
                                      "The source (infecting) compartment", coupling0, "and first affected compartment (here: ", affected0,
                                      "), must be equal on both sides of the reaction equation but are not.",
                                      )

                if self.get_compartment_id(coupling0) != event[_AFFECTED_TARGET_COMPARTMENT]:
                    raise ValueError("In conditional link transmission process",
                                      coupling0, coupling1, "->", affected0, affected1,
                                      "The source (infecting) compartment", coupling0, "must be equal to the affected (target) compartment of",
                                      "the triggering event", triggering_event, " but is not."
                                      )

                _c0 = self.get_compartment_id(coupling0)
                _s = self.get_compartment_id(coupling1)
                _t = self.get_compartment_id(affected1)
                
                this_triggered_event = (_c0, _s, _t)
                self.conditional_link_transmission_events[event][_s] = this_triggered_event

        return self

    def _zip_events(self):
        """
        Concatenate link and node events.
        """
        self.node_and_link_events = [ None for n in range(self.N_comp) ]

        # for each compartment, concatenate node and link event lists
        for comp in range(self.N_comp):
            link_events = self.link_transmission_events[comp]
            node_events = self.node_transition_events[comp]

            # the last entry in this list saves the event index range that are link events.
            # we need this, because for every node, these rates have to be multiplied by
            # the out degree later on.
            if len(node_events[_EVENTS]) == 0 and len(link_events[_EVENTS]) == 0:
                events = np.array([])
                rates = np.array([])
                indices = [0, 0]
            elif len(node_events[_EVENTS]) != 0 and len(link_events[_EVENTS]) == 0:
                events = node_events[_EVENTS]
                rates = node_events[_RATES]
                indices = [0, 0]
            elif len(node_events[_EVENTS]) == 0 and len(link_events[_EVENTS]) != 0:
                events = link_events[_EVENTS]
                rates = link_events[_RATES]
                indices = [0, len(link_events[_EVENTS])]
            else:
                events = np.concatenate((node_events[_EVENTS],link_events[_EVENTS]))
                rates = np.concatenate((node_events[_RATES],link_events[_RATES]))
                indices = [len(node_events[_RATES]), len(node_events[_RATES]) + len(link_events[_RATES])]

            self.node_and_link_events[comp] = [ events, rates, indices ]

    def set_random_initial_conditions(self, initial_conditions):
        """
        Set random initial conditions for each compartment.

        Parameters
        ----------
        initial_conditions : dict
            A dictionary that maps a compartment string to a number of nodes
            that should be sampled to be in this compartment initially.
        """

        if type(initial_conditions) == dict:
            initial_conditions = list(initial_conditions.items())

        # initially, all nodes can be distributed
        nodes_left = set(range(self.N_nodes))

        # save node status and initial conditions
        node_status = np.zeros(self.N_nodes,dtype=int)
        y0 = np.zeros(self.N_comp)
        total = 0

        # iterate through all initial conditions
        for compartment, amount in initial_conditions:

            # count the number of nodes in this compartment
            total += amount
            comp_id = self.get_compartment_id(compartment)

            # check whether it was defined before
            if y0[comp_id] != 0:
                raise ValueError('Double entry in initial conditions for compartment '+str(compartment))
            else:
                # if node, sample `amount` nodes from the remaining set,
                # take those nodes out of the set and carry on with the set of remaining nodes
                these_nodes = np.random.choice(list(nodes_left),size=amount,replace=False)
                y0[comp_id] = amount
                node_status[these_nodes] = comp_id
                nodes_left = nodes_left.difference(these_nodes)

        # check that the initial conditions contain all nodes
        if np.abs(total-self.N_nodes)/self.N_nodes > 1e-14:
            raise ValueError('Sum of initial conditions does not equal unity.')

        self.set_node_statuses(node_status)
        self.y0 = y0

        return self

    def set_node_statuses(self,node_status):
        """
        Set all node statuses at once and evaluate events and rates accordingly.
        Can be used to set initial conditions.

        Parameters
        ----------
        node_status : numpy.ndarray of int
            For each node, this array contains the node's compartment index. 
        """

        if len(node_status) != self.N_nodes:
            raise ValueError("`node_status` must carry N_nodes values")

        # filter out the minimally possible rate (other than zero) and the maximally
        # possible rate 
        kmax = self.out_degree.max()
        kmin = self.out_degree.min()
        compartment_mins = np.zeros(self.N_comp)
        compartment_maxs = np.zeros(self.N_comp)
        for compartment, (event, rates) in enumerate(self.node_transition_events):
            for rate in rates:
                compartment_mins[compartment] += rate
                compartment_maxs[compartment] += rate
        for compartment, (event, rates) in enumerate(self.link_transmission_events):
            for rate in rates:
                compartment_mins[compartment] += rate * kmin
                compartment_maxs[compartment] += rate * kmax

        compartment_min = (compartment_mins[compartment_mins>0]).min()
        compartment_max = (compartment_maxs).max()

        # set these node statuses
        self.node_status = node_status.copy()    

        # construct a SamplableSet in which we insert the node reaction rates
        self.all_node_events = SamplableSet(compartment_min,compartment_max,cpp_type='int')

        # construct a list that contains events and rates for each node once it was triggered
        self.node_event_probabilities = [ [] for n in range(self.N_nodes) ]

        # set each node status according to the passed node_status array
        for node, status in enumerate(self.node_status):
            self.set_node_status(node, status)

        return self

    def set_node_status(self,node,status):
        """
        Set the status of node `node` to `status`

        Parameters
        ----------
        node : int
            The index of the node 
        status : int
            The index of the compartment that the node changes into.
        """

        # get the events that can happen to a node of this status/compartment
        status_events = self.node_and_link_events[status]

        # if no events can happen, remove the node from 
        # the event set and set the node events to an empty set
        if len(status_events[_RATES]) == 0:
            del self.all_node_events[node]
            self.node_event_probabilities[node] = []
        else:
            # otherwise, get a copy of this compartment's rate vector
            these_rates = status_events[_RATES].copy()
            # scale the link events by this node's out degree            
            scale_from, scale_to = status_events[_LINK_PROCESS_INDICES]
            these_rates[scale_from:scale_to] *= self.out_degree[node]
            total_rate = these_rates.sum()
            # add the node to the node event set 
            self.all_node_events[node] = total_rate
            # save the probabilities with which each of the node's events happens
            self.node_event_probabilities[node] = these_rates / total_rate

        # finally, set node status
        self.node_status[node] = status

        return self

    def get_total_event_rate(self):
        """
        Get the total event rate.
        """
        return self.all_node_events.total_weight()

    def get_reacting_node(self):
        """
        Get a reacting node with probability proportional to its reaction rate.
        """
        return self.all_node_events.sample()[0]

    def make_random_node_event(self, reacting_node):
        """
        Let a random node event happen according to the event probabilities
        of this node's status.

        Parameters
        ----------
        reacting_node : int
            the index of the node that reacts

        Returns
        -------
        compartment_changes : list of tuples of int
            Each tuple contains the index of
            the compartment that loses a member on the first position
            and the index of the compartment that gains a member.
        """

        # once a node that reacts, obtain the probabilities with which
        # each of its events happens, and its current status
        status = self.node_status[reacting_node]
        node_event_probabilities = self.node_event_probabilities[reacting_node]

        # sample the index of the event that happens according to the node probabilities
        # and get the corresponding event
        event_index = np.random.choice(len(node_event_probabilities),p=node_event_probabilities)
        event = self.node_and_link_events[status][_EVENTS][event_index]

        return self.make_node_event(reacting_node, tuple(event))

    def make_node_event(self, reacting_node, event, neighbor=None):
        """
        Let a specific node event happen

        Parameters
        ----------
        reacting_node : int
            the index of the node that reacts
        event : tuple of int
            three-entry long tuple that characterizes this event
        neighbor : int, default = None
            specify the neighbor to which this specific event happens

        Returns
        -------
        compartment_changes : list of tuples of int
            Each tuple contains the index of
            the compartment that loses a member on the first position
            and the index of the compartment that gains a member.
        """

        # classify the event and changing compartments
        is_transmission_event = event[_TRANSMITTING_COMPARTMENT] != -1
        losing_compartment = event[_AFFECTED_SOURCE_COMPARTMENT]
        gaining_compartment = event[_AFFECTED_TARGET_COMPARTMENT]

        compartment_changes = []

        # if is transmission event
        if is_transmission_event:            
            if neighbor is None:
                # if no reacting neighbor was specified    
                if self.is_network_model:
                    # sample a neighbor from the reacting node's neighbor set
                    neighbor, _ = self.graph[reacting_node].sample()
                else:
                    # in well-mixed, sample any other node
                    neighbor = np.random.choice(self.N_nodes-1)
                    if neighbor >= reacting_node:
                        neighbor += 1
            # if the sampled neighbor is of the affected compartment,
            # the affected node is the neighbor
            if self.node_status[neighbor] == event[_AFFECTED_SOURCE_COMPARTMENT]:
                affected_node = neighbor
            else:
                # otherwise, nothing happens
                return compartment_changes
        else:
            # if it's a node transition event, this node is affected
            affected_node = reacting_node

        new_compartment = event[_AFFECTED_TARGET_COMPARTMENT]
        self.set_node_status(affected_node,new_compartment)

        # check for conditional processes based on this process
        if self.conditional_link_transmission_events is not None:

            # check whether an entry exists for this event
            try:
                conditional_events = self.conditional_link_transmission_events[event]
                has_conditional_transmissions = True
            except KeyError as e:
                has_conditional_transmissions = False

            # if it exists
            if has_conditional_transmissions:

                # sample neighbors of the affected node (might include this reacting node)
                if self.is_network_model:
                    neighbors = self.graph[affected_node]
                else:
                    neighbors = np.random.choice(self.N_nodes-1,size=int(self.out_degree[affected_node]),replace=False)
                    neighbors[neighbors >= affected_node] += 1

                # iterate through neighbors
                for n in neighbors:
                    # samplableset returns node ids together with weight, therefore we 
                    # have to ask for the first entry of n
                    try:
                        n = n[0]
                    except IndexError as e:
                        pass

                    # if the compartment of the neighbor is affected by a conditional event,
                    # let this conditional event happen to the neighbor, as triggered
                    # by the affected node
                    try:
                        conditional_transmission_event = conditional_events[self.node_status[n]]
                        these_compartment_changes = self.make_node_event(affected_node, conditional_transmission_event, neighbor = n)
                        compartment_changes.extend(these_compartment_changes)
                    except KeyError as e:
                        pass

        compartment_changes.append((losing_compartment, gaining_compartment))

        return compartment_changes

    def simulation(self,tmax,return_compartments=None,sampling_dt=None):
        """
        Returns values of the given compartments at the demanded
        time points (as a numpy.ndarray of shape 
        ``(return_compartments), len(time_points)``.

        If ``return_compartments`` is None, all compartments will
        be returned.
        """

        if return_compartments is None:
            return_compartments = self.compartments

        ndx = [self.get_compartment_id(C) for C in return_compartments]
        current_state = self.y0.copy()
        compartments = [ current_state.copy() ]

        t = 0.0
        time = [0.0]
        while t < tmax and self.get_total_event_rate() > 0:

            # sample and advance time according to current total rate
            tau = np.random.exponential(1/self.get_total_event_rate())
            new_t = t+tau

            # break if simulation time is reached
            if new_t >= tmax:
                break

            # sample a reacting node from the reaction set
            reacting_node = self.get_reacting_node()

            # let the event take place and get all the compartments
            # that are associated with changes
            changing_compartments = self.make_random_node_event(reacting_node)

            # only save compartment counts if anything changed
            if len(changing_compartments) > 0:

                if sampling_dt is not None:
                    # sample all the time steps that were demanded in between the two events
                    last_sample_dt = time[-1]
                    for idt in range(1,int(np.ceil((new_t-last_sample_dt)/sampling_dt))):
                        time.append(last_sample_dt+idt*sampling_dt)
                        compartments.append(current_state.copy())

                # write losses and gains into the current state vector
                for losing, gaining in changing_compartments:
                    current_state[losing] -= 1 
                    current_state[gaining] += 1 

                # save the current state if sampling_dt wasn't specified
                if sampling_dt is None:
                    time.append(t)
                    compartments.append(current_state.copy())


            # advance time
            t = new_t


        # convert to result dictionary
        time = np.array(time)
        result = np.array(compartments)

        return time, { compartment: result[:,c_ndx] for c_ndx, compartment in zip(ndx, return_compartments) }

if __name__ == "__main__":

    model = StochasticEpiModel(list("SI"),3)

    model.set_node_transition_processes([("I",1.0,"S")])
    model.set_link_transmission_processes([("I","S",2.0,"I","I")])

    print(model.link_transmission_events)
    print(model.node_transition_events)
    for comp, comp_events in enumerate(model.node_and_link_events):
        print(model.get_compartment(comp), comp_events)
    
    #model.set_well_mixed(3,mean_contact_number=1)
    model.set_random_initial_conditions({"S": 2, "I": 1})

    print(model.y0)
    print(model.node_status)

    reacting_node = model.get_reacting_node()

    print("reacting node:", reacting_node)

    model.make_random_node_event(reacting_node)

    print(model.node_status)


    print("===================")


    N = 10000
    i0 = 0.05
    I0 = int(i0*N)
    S0 = N - I0

    R0 = 2.0
    mu = 1.0
    k0 = 100

    model = StochasticEpiModel(list("SI"),N,well_mixed_mean_contact_number=k0)

    model.set_node_transition_processes([("I",mu,"S")],)
    model.set_link_transmission_processes([("I","S",R0/k0,"I","I")])
    model.set_random_initial_conditions({"S":S0,"I":I0})

    t, result = model.simulation(4,sampling_dt = 0.1)

    from bfmplot import pl
    for comp, series in result.items():
        pl.plot(t, series, label=comp)

    from metapop.epi import SISModel

    model = SISModel(R0=R0,recovery_rate=mu,population_size=N)
    model.set_initial_conditions({"S":S0,"I":I0})
    result = model.get_result_dict(t)

    for comp, series in result.items():
        pl.plot(t, series, label=comp)

    pl.legend()
    print("===================")


    N = 10000
    i0 = 0.05
    I0 = int(i0*N)
    S0 = N - I0

    R0 = 2.0
    mu = 1.0
    k0 = 100

    model = StochasticEpiModel(list("SIR"),N,well_mixed_mean_contact_number=k0)

    model.set_node_transition_processes([
                                            ("I",mu,"R"),
                                        ])
    model.set_link_transmission_processes([("I","S",R0/k0*mu,"I","I")])
    model.set_random_initial_conditions({"S":S0,"I":I0})

    t, result = model.simulation(4,sampling_dt = 0.1)

    pl.figure()

    for comp, series in result.items():
        pl.plot(t, series, label=comp)

    from metapop.epi import SIRModel

    model = SIRModel(R0=R0,recovery_rate=mu,population_size=N)
    model.set_initial_conditions({"S":S0,"I":I0})
    result = model.get_result_dict(t)

    for comp, series in result.items():
        pl.plot(t, series, label=comp)
    
    pl.legend()

    print("====== BARABASI-ALBERT ========")


    N = 1000
    i0 = 0.05
    I0 = int(i0*N)
    S0 = N - I0

    R0 = 2.0
    mu = 1.0
    k0 = 100

    import networkx as nx
    G = nx.barabasi_albert_graph(N,k0)
    weighted_edge_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(weighted_edge_tuples) / N

    model = StochasticEpiModel(list("SIR"),N,edge_weight_tuples=weighted_edge_tuples)

    model.set_node_transition_processes([
                                            ("I",mu,"R"),
                                        ])
    model.set_link_transmission_processes([("I","S",R0/k_norm*mu,"I","I")])
    model.set_network(N,weighted_edge_tuples)
    model.set_random_initial_conditions({"S":S0,"I":I0})

    from time import time
    start = time()
    t, result = model.simulation(100)
    end = time()
    print("composition rejection needed", end-start,"seconds") 


    pl.figure()

    for comp, series in result.items():
        pl.plot(t, series, label=comp)

    from tacoma.epidemics import SIR_weighted

    start = time()
    model = SIR_weighted(N,weighted_edge_tuples,infection_rate=R0/k_norm*mu,recovery_rate=mu,number_of_initially_infected=I0)
    end = time()
    print("force of infection needed", end-start,"seconds") 
    _t, _I, _R = model.simulation(100)

    pl.plot(_t, _I, label="I")
    pl.plot(_t, _R, label="R")
    

    pl.legend()
    pl.show()
