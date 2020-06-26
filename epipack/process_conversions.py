
def processes_to_rates(process_list, compartments):
    """
    Converts a list of reaction process tuples to rate tuples

    Parameters
    ----------
    process_list : :obj:`list` of :obj:`tuple`
        A list containing reaction processes in terms of tuples.

        .. code:: python

            [
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
            ]

    compartments : :obj:`list` of hashable type
        The compartments of these reaction equations. 

    Returns
    -------
    quadratic_rates : :obj:`list` of :obj:`tuple`
        Rate tuples for quadratic terms
    linear_rates : :obj:`list` obj:`tuple`
        Rate tuples for linear terms
    """

    quadratic_rates = []
    linear_rates = []

    for process in process_list:

        if len(process) == 3:
            # it's either a transition process or a birth process:
            if process[1] not in compartments:
                # it's a fission process
                linear_rates.extend(transition_processes_to_rates([process]))
            else:
                raise TypeError("Process " + str(process) + " is not understood.")

        elif len(process) == 4:
            # it's either a fission process or a fusion process

            if process[1] not in compartments:
                # it's a fission process
                quadratic_rates.extend(fission_processes_to_rates([process]))
            elif process[2] not in compartments:
                # it's a fusion process
                quadratic_rates.extend(fusion_processes_to_rates([process]))
            else:
                raise TypeError("Process " + str(process) + " is not understood.")

        elif len(process) == 5:

            # it's a transmission process
            if process[2] not in compartments:
                quadratic_rates.extend(transmission_processes_to_rates([process]))
            else:
                raise TypeError("Process " + str(process) + " is not understood.")

        else:
            raise TypeError("Process " + str(process) + " is not understood.")

    return quadratic_rates, linear_rates


def transition_processes_to_rates(process_list):
    """
    Define the transition processes between compartments, including birth and deaths processes.

    Parameters
    ==========

    process_list : :obj:`list` of :obj:`tuple`
        A list of tuples that contains transitions rates in the following format:

        .. code:: python

            [
                ( source_compartment, rate, target_compartment ),
                ...
            ]

    Example
    -------

    For an SEIR model.

    .. code:: python

        transition_processes_to_rates([
            ("E", symptomatic_rate, "I" ),
            ("I", recovery_rate, "R" ),
        ])

    """
    linear_rates = []

    for source, rate, target in process_list:

        if source is None and target is None:
            raise ValueError("The reaction" + str((source, rate, target)) + " is meaningless because there are no reactants.")
        elif source is None and target is not None:
            #birth process
            linear_rates.append( (None, target, rate) )
        elif source is not None and target is None:
            #death process
            linear_rates.append( (source, source, -rate) )
        else:
            # source compartment loses an entity
            # target compartment gains one
            linear_rates.append((source, source, -rate))
            linear_rates.append((source, target, +rate))

    return linear_rates


def fission_processes_to_rates(process_list):
    """
    Define linear fission processes between compartments.

    Parameters
    ==========

    process_list : :obj:`list` of :obj:`tuple`
        A list of tuples that contains fission rates in the following format:

        .. code:: python

            [
                ("source_compartment", rate, "target_compartment_0", "target_compartment_1" ),
                ...
            ]

    Example
    -------

    For pure exponential growth of compartment `B`.

    .. code:: python

        epi.set_fission_processes([
            ("B", growth_rate, "B", "B" ),
        ])

    """

    linear_rates = []
    
    for source, rate, target0, target1 in process_list:

        _s = source
        _t0 = target0
        _t1 = target1
        
        # source compartment loses an entity
        # target compartment gains one
        linear_rates.append((_s, _s, -rate))
        linear_rates.append((_s, _t0, +rate))
        linear_rates.append((_s, _t1, +rate))

    return linear_rates

def fusion_processes_to_rates(process_list):
    """
    Define fusion processes between compartments.

    Parameters
    ==========

    process_list : :obj:`list` of :obj:`tuple`
        A list of tuples that contains fission rates in the following format:

        .. code:: python

            [
                ("coupling_compartment_0", "coupling_compartment_1", rate, "target_compartment_0" ),
                ...
            ]

    Example
    -------

    Fusion of reactants "A", and "B" to form "C".

    .. code:: python

        fusion_processes_to_rates([
            ("A", "B", reaction_rate, "C" ),
        ])

    """

    quad_rates = []
    
    for source0, source1, rate, target in process_list:
        quad_rates.append((source0, source1, target, rate))
        quad_rates.append((source0, source1, source0, -rate))
        quad_rates.append((source0, source1, source1, -rate))

    return quad_rates

def transmission_processes_to_rates(process_list):
    r"""
    A wrapper to define quadratic process rates through transmission reaction equations.
    Note that in stochastic network/agent simulations, the transmission
    rate is equal to a rate per link. For the mean-field ODEs,
    the rates provided to this function will just be equal 
    to the prefactor of the respective quadratic terms.

    For instance, if you analyze an SIR system and simulate on a network of mean degree :math:`k_0`,
    a basic reproduction number :math:`R_0`, and a recovery rate :math:`\mu`,
    you would define the single link transmission process as 

        .. code:: python

            ("I", "S", R_0/k_0 * mu, "I", "I")

    For the mean-field system here, the corresponding reaction equation would read
        
        .. code:: python
            
            ("I", "S", R_0 * mu, "I", "I")

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

        transmission_processes_to_rates([
            ("I", "S", +1, "I", "E" ),
        ])

    """

    # each compartment is associated with a list of events that it can be
    # involved in (event = first entry)
    # Each event is associated with a rate (rate = second entry)
    rate_list = []

    # iterate through processes
    for coupling0, coupling1, rate, affected0, affected1 in process_list:

        #if coupling0 != affected0:
        #    raise ValueError(" ".join("In process ",
        #                      coupling0, coupling1, " -> ", affected0, affected1,
        #                      "The source (infecting) compartment", coupling0, "and first affected compartment (here: ", affected0,
        #                      "), must be equal on both sides of the reaction equation but are not.",
        #                      ))


        _s0 = coupling0
        _s1 = coupling1
        _t0 = affected0
        _t1 = affected1

        reactants = [_s0, _s1]
        products = [_t0, _t1]
        constant = (set.intersection(set(reactants), set(products)))        
        if len(constant) == 2 or tuple(reactants) == tuple(products):
            raise ValueError("Process "+\
                             str((coupling0, coupling1, rate, affected0, affected1)) +\
                             " leaves system unchanged")
        elif len(constant) == 1:
            #print("reactants", reactants)
            #print("constant", constant)
            #print("products", products)
            constant = next(iter(constant))
            reactants.remove(constant)
            products.remove(constant)
            _s0 = constant
            _s1 = reactants[0]
            _t1 = products[0]

            rate_list.append( (_s0, _s1, _s1, -rate) )
            rate_list.append( (_s0, _s1, _t1, +rate) )
            #print(rate_list)
        else:

            rate_list.append( (_s0, _s1, _s1, -rate) )
            rate_list.append( (_s0, _s1, _t1, +rate) )
            rate_list.append( (_s0, _s1, _s0, -rate) )
            rate_list.append( (_s0, _s1, _t0, +rate) )
        #rate_list.append( (_s0, _s1, _s1, -rate) )
        #rate_list.append( (_s0, _s1, _t1, +rate) )
        #rate_list.append( (_s0, _s1, _s0, -rate) )
        #rate_list.append( (_s0, _s1, _t0, +rate) )

    return rate_list