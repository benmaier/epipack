"""
Provides an API to define  epidemiological models.
"""

import numpy as np
import scipy.sparse as sprs

import warnings

from epipack.integrators import (
        IntegrationMixin,
        time_leap_newton,
        time_leap_ivp,
    )

from epipack.process_conversions import (
        processes_to_events,
        transition_processes_to_events,
        fission_processes_to_events,
        fusion_processes_to_events,
        transmission_processes_to_events,
    )

from scipy.optimize import newton
from scipy.integrate import quad

def custom_choice(p):
    """
    Return an index of normalized probability
    vector ``p`` with probability equal to
    the corresponding entry in ``p``.
    """
    return np.argmin(np.cumsum(p)<np.random.rand())

class ConstantBirthRate():
    """
    Will be used as a function of
    time ``t`` and state ``y``,
    returning a rate value.

    Parameters
    ----------
    rate : float
        Constant rate value
    """
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, t, y):
        return self.rate

class DynamicBirthRate():
    """
    Will be used as a function of
    time ``t`` and state ``y``,
    returning a rate value.

    Parameters
    ----------
    rate : function
        Function of time ``t`` and state ``y``
    """

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, t, y):
        return self.rate(t,y)

class ConstantLinearRate:
    """
    Will be used as a function of
    time ``t`` and state ``y``,
    returning a rate value.

    Parameters
    ----------
    rate : float
        Constant rate value
    comp0 : int
        Index of the corresponding reacting
        component. The incidence of this component
        will be multiplied with the
        value of ``rate``.
    """

    def __init__(self, rate, comp0):
        self.rate = rate
        self.comp0 = comp0

    def __call__(self, t, y):
        return self.rate * y[self.comp0]

class DynamicLinearRate:
    """
    Will be used as a function of
    time ``t`` and state ``y``,
    returning a rate value.

    Parameters
    ----------
    rate : function
        Function of time ``t`` and state ``y``
    comp0 : int
        Index of the corresponding reacting
        component. The incidence of this component
        will be multiplied with the
        value of ``rate``.
    """

    def __init__(self, rate, comp0):
        self.rate = rate
        self.comp0 = comp0

    def __call__(self, t, y):
        return self.rate(t,y) * y[self.comp0]

class ConstantQuadraticRate:
    """
    Will be used as a function of
    time ``t`` and state ``y``,
    returning a rate value.

    Parameters
    ----------
    rate : float
        Constant rate value
    comp0 : int
        Index of one of the reacting
        components. The incidence of this component
        will be multiplied with the
        value of ``rate``.
    comp1 : int
        Index of the other reacting
        component. The incidence of this component
        will be multiplied with the
        value of ``rate``.
    """

    def __init__(self, rate, comp0, comp1):
        self.rate = rate
        self.comp0 = comp0
        self.comp1 = comp1

    def __call__(self, t, y):
        return self.rate * y[self.comp0] * y[self.comp1]

class DynamicQuadraticRate:
    """
    Will be used as a function of
    time ``t`` and state ``y``,
    returning a rate value.

    Parameters
    ----------
    rate : function
        Function of time ``t`` and state ``y``
    comp0 : int
        Index of one of the reacting
        components. The incidence of this component
        will be multiplied with the
        value of ``rate``.
    comp1 : int
        Index of the other reacting
        component. The incidence of this component
        will be multiplied with the
        value of ``rate``.
    """

    def __init__(self, rate, comp0, comp1):
        self.rate = rate
        self.comp0 = comp0
        self.comp1 = comp1

    def __call__(self, t, y):
        return self.rate(t,y) * y[self.comp0] * y[self.comp1]

class EpiModel(IntegrationMixin):
    """
    A general class to define a standard
    mean-field compartmental
    epidemiological model, based on reaction
    events.

    Parameters
    ----------
    compartments : :obj:`list` of :obj:`string`
        A list containing compartment strings.
    initial_population_size : float, default = 1.0
        The population size at :math:`t = 0`.
    correct_for_dynamical_population_size : bool, default = False
        If ``True``, the quadratic coupling terms will be
        divided by the population size.
    integral_solver : str, default = 'solve_ivp'
        Whether or not to use the initial-value solver ``solve_ivp``.
        to determine a time leap for time-varying rates.
        If not ``'solve_ivp'``, a Newton-Raphson method will be
        used on the upper bound of a quad-integrator.

    Attributes
    ----------
    compartments : list
        A list containing strings or hashable types that describe each compartment,
        (e.g. "S", "I", etc.).
    compartment_ids: dict
        Maps compartments to their indices in ``self.compartments``.
    N_comp : :obj:`int`
        Number of compartments (including population number)
    initial_population_size : float
        The population size at :math:`t = 0`.
    correct_for_dynamical_population_size : bool
        If ``True``, the quadratic coupling terms will be
        divided by the sum of all compartments, otherwise they
        will be divided by the initial population size.
    birth_rate_functions : list of ConstantBirthRate or DynamicBirthRate
        A list of functions that return rate values based on time ``t``
        and state vector ``y``. Each entry corresponds to an event update
        in ``self.birth_event_updates``.
    birth_event_updates : list of numpy.ndarray
        A list of vectors. Each entry corresponds to a rate in
        ``birth_rate_functions`` and quantifies the change in
        individual counts in the compartments.
    linear_rate_functions : list of ConstantLinearRate or DynamicLinearRate
        A list of functions that return rate values based on time ``t``
        and state vector ``y``. Each entry corresponds to an event update
        in ``self.linear_event_updates``.
    linear_event_updates : list of numpy.ndarray
        A list of vectors. Each entry corresponds to a rate in
        ``linear_rate_functions`` and quantifies the change in
        individual counts in the compartments.
    quadratic_rate_functions : list of ConstantQuadraticRate or DynamicQuadraticRate
        A list of functions that return rate values based on time ``t``
        and state vector ``y``. Each entry corresponds to an event update
        in ``self.quadratic_event_updates``.
    quadratic_event_updates : list of numpy.ndarray
        A list of vectors. Each entry corresponds to a rate in
        ``quadratic_rate_functions`` and quantifies the change in
        individual counts in the compartments.
    y0 : numpy.ndarray
        The initial conditions.
    rates_have_explicit_time_dependence : bool
        Internal switch that's flipped when a non-constant
        rate is passed to the model.
    use_ivp_solver : bool
        Whether or not to use the initial-value solver
        to determine a time leap for time-varying rates.
        If ``False``, a Newton-Raphson method will be
        used on the upper bound of a quad-integrator.

    Example
    -------

    .. code:: python

        >>> epi = EpiModel(["S","I","R"])
        >>> print(epi.compartments)
        [ "S", "I", "R" ]


    """

    def __init__(self,
                      compartments,
                      initial_population_size=1,
                      correct_for_dynamical_population_size=False,
                      integral_solver='solve_ivp',
                  ):

        self.y0 = None

        self.compartments = list(compartments)
        self.compartment_ids = { C: iC for iC, C in enumerate(compartments) }
        self.N_comp = len(self.compartments)

        self.initial_population_size = initial_population_size
        self.correct_for_dynamical_population_size = correct_for_dynamical_population_size

        self.birth_rate_functions = []
        self.birth_event_updates = []
        self.linear_rate_functions = []
        self.linear_event_updates = []
        self.quadratic_rate_functions = []
        self.quadratic_event_updates = []

        self.birth_events = []
        self.linear_events = []
        self.quadratic_events = []

        self.use_ivp_solver = integral_solver == 'solve_ivp'

        self.rates_have_explicit_time_dependence = False

    def get_compartment_id(self,C):
        """Get the integer ID of a compartment ``C``"""
        return self.compartment_ids[C]

    def get_compartment(self,iC):
        """Get the compartment, given an integer ID ``iC``"""
        return self.compartments[iC]

    def set_processes(self,
                      process_list,
                      allow_nonzero_column_sums=False,
                      reset_events=True,
                      ignore_rate_position_checks=False):
        """
        Converts a list of reaction process tuples to event tuples and sets the rates for this model.

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
        allow_nonzero_column_sums : bool, default : False
            Traditionally, epidemiological models preserve the
            total population size. If that's not the case,
            switch off testing for this.
        reset_events : bool, default : True
            If this is `True`, reset all events to zero before setting the new ones.
        ignore_rate_position_checks : bool, default = False
            This function usually checks whether the rate of
            a reaction is positioned correctly. You can
            turn this behavior off for transition, birth, death, and
            transmission processes. (Useful if you want to define
            symbolic transmission processes that are compartment-dependent).
        """

        quadratic_events, linear_events = processes_to_events(process_list,
                                                              self.compartments,
                                                              ignore_rate_position_checks)
        self.set_linear_events(linear_events,
                               allow_nonzero_column_sums=allow_nonzero_column_sums,
                               reset_events=reset_events,
                               )
        self.set_quadratic_events(quadratic_events,
                                  allow_nonzero_column_sums=allow_nonzero_column_sums,
                                  reset_events=reset_events,
                                  )

        return self

    def _rate_has_functional_dependency(self,rate):
        if callable(rate):
            t = [0,1,2,10000,-10000]
            y = np.ones(self.N_comp)
            test = np.array([ rate(_t, y) for _t in t ])
            has_time_dependence = not np.all(test == test[0])
            self.rates_have_explicit_time_dependence = \
                    self.rates_have_explicit_time_dependence or has_time_dependence
            return True
        else:
            return False

    def set_linear_events(self,
                          event_list,
                          allow_nonzero_column_sums=False,
                          reset_events=True):
        r"""
        Define the linear transition events between compartments.

        Parameters
        ==========
        event_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transition events in the
            following format:

            .. code:: python

                [
                    (
                        ("affected_compartment_0",),
                        rate,
                        [
                            ("affected_compartment_0", dN0),
                            ("affected_compartment_1", dN1),
                            ...
                        ],
                     ),
                    ...
                ]

        allow_nonzero_column_sums : :obj:`bool`, default : False
            Traditionally, epidemiological models preserve the
            total population size. If that's not the case,
            switch off testing for this.
        reset_events : bool, default : True
            Whether to reset all linear events to zero before
            converting those.

        Example
        -------
        For an SEIR model with infectious period ``tau``
        and incubation period ``theta``.

        .. code:: python

            epi.set_linear_events([
                ( ("E",),
                  1/theta,
                  [ ("E", -1), ("I", +1) ]
                ),
                ( ("I",),
                  1/tau,
                  [ ("I", -1), ("R", +1) ]
                ),
            ])

        Read as "compartment E reacts with rate :math:`1/\theta`
        which leads to the decay of one E particle to one I particle."
        """

        if reset_events:
            birth_rate_functions = []
            birth_event_updates = []
            linear_rate_functions = []
            linear_event_updates = []
            birth_events = []
            linear_events = []
        else:
            linear_event_updates = list(self.linear_event_updates)
            birth_event_updates = list(self.birth_event_updates)
            linear_rate_functions = list(self.linear_rate_functions)
            birth_rate_functions = list(self.birth_rate_functions)
            birth_events = list(self.birth_events)
            linear_events = list(self.linear_events)


        for acting_compartments, rate, affected_compartments in event_list:

            dy = np.zeros(self.N_comp)
            for trg, change in affected_compartments:
                _t = self.get_compartment_id(trg)
                dy[_t] += change

            if acting_compartments[0] is None:
                if self._rate_has_functional_dependency(rate):
                    this_rate = DynamicBirthRate(rate)
                else:
                    this_rate = ConstantBirthRate(rate)
                birth_event_updates.append( dy )
                birth_rate_functions.append( this_rate )
                birth_events.append((acting_compartments, rate, affected_compartments))
            else:
                _s = self.get_compartment_id(acting_compartments[0])
                if self._rate_has_functional_dependency(rate):
                    this_rate = DynamicLinearRate(rate, _s)
                else:
                    this_rate = ConstantLinearRate(rate, _s)
                linear_event_updates.append( dy )
                linear_rate_functions.append( this_rate )
                linear_events.append((acting_compartments, rate, affected_compartments))

            if dy.sum() != 0 and not self.correct_for_dynamical_population_size:
                warnings.warn("This model has processes with a fluctuating "+\
                        "number of agents. Consider correcting the rates dynamically with "+\
                        "the attribute correct_for_dynamical_population_size = True")


        if not allow_nonzero_column_sums and len(linear_rate_functions)>0:
            _y = np.ones(self.N_comp)
            test = sum([r(0,_y) * dy for dy, r in zip (linear_event_updates, linear_rate_functions)])
            test += sum([r(0,_y) * dy for dy, r in zip (birth_event_updates, birth_rate_functions)])
            test_sum = test.sum()
            if np.abs(test_sum) > 1e-15:
                warnings.warn("events do not sum to zero for each column:" + str(test_sum))

        self.linear_event_updates = linear_event_updates
        self.linear_rate_functions = linear_rate_functions
        self.birth_event_updates = birth_event_updates
        self.birth_rate_functions = birth_rate_functions
        self.linear_events = linear_events
        self.birth_events = birth_events

        return self


    def add_transition_processes(self,process_list):
        """
        Define the linear transition processes between compartments.

        Parameters
        ==========
        process_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions events in the following format:

            .. code:: python

                [
                    ( source_compartment, rate, target_compartment ),
                    ...
                ]

        Example
        =======

        For an SEIR model.

        .. code:: python

            epi.add_transition_processes([
                ("E", symptomatic_rate, "I" ),
                ("I", recovery_rate, "R" ),
            ])

        """

        linear_events = transition_processes_to_events(process_list)

        return self.set_linear_events(linear_events,
                                      reset_events=False,
                                      allow_nonzero_column_sums=True)

    def add_fission_processes(self,process_list):
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

            epi.add_fission_processes([
                ("B", growth_event, "B", "B" ),
            ])

        """
        linear_events = fission_processes_to_events(process_list)

        return self.set_linear_events(linear_events,
                                      reset_events=False,
                                      allow_nonzero_column_sums=True)

    def add_fusion_processes(self,process_list):
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

            epi.add_fusion_processes([
                ("A", "B", reaction_rate, "C" ),
            ])

        """
        quadratic_events = fusion_processes_to_events(process_list)

        return self.set_quadratic_events(quadratic_events,
                                         reset_events=False, 
                                         allow_nonzero_column_sums=True)

    def add_transmission_processes(self,process_list):
        r"""
        A wrapper to define quadratic process rates
        through transmission reaction equations.
        Note that in stochastic network/agent simulations, the transmission
        rate is equal to a rate per link. For the mean-field ODEs,
        the rates provided to this function will just be equal 
        to the prefactor of the respective quadratic terms.

        For instance, if you analyze an SIR system and simulate
        on a network of mean degree :math:`k_0`,
        a basic reproduction number :math:`R_0`, and a 
        recovery rate :math:`\mu`, you would define the single 
        link transmission process as 

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

            epi.add_transmission_processes([
                ("I", "S", +1, "I", "E" ),
            ])

        """
        quadratic_events = transmission_processes_to_events(process_list)

        return self.set_quadratic_events(quadratic_events,
                                         reset_events=False, 
                                         allow_nonzero_column_sums=True)

    def add_quadratic_events(self,
                             event_list,
                             allow_nonzero_column_sums=False):
        """
        Add quadratic events without resetting the existing event terms.
        See :func:`epipack.numeric_epi_models.EpiModel.set_quadratic_events` for docstring.
        """

        return self.set_quadratic_events(event_list,
                                         reset_events=False,
                                         allow_nonzero_column_sums=allow_nonzero_column_sums,
                                         )

    def add_linear_events(self,
                          event_list,
                          allow_nonzero_column_sums=False):
        """
        Add linear events without resetting the existing event terms.
        See :func:`epipack.numeric_epi_models.EpiModel.set_linear_events` for docstring.
        """

        return self.set_linear_events(event_list,
                                      reset_events=False,
                                      allow_nonzero_column_sums=allow_nonzero_column_sums
                                      )

    def set_quadratic_events(self,
                             event_list,
                             allow_nonzero_column_sums=False,
                             reset_events=True,
                             ):
        r"""
        Define quadratic transition events between compartments.

        Parameters
        ----------
        event_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transmission events in the following format:

            .. code:: python

                [
                    (
                        ("coupling_compartment_0", "coupling_compartment_1"),
                        rate,
                        [
                            ("affected_compartment_0", dN0),
                            ("affected_compartment_1", dN1),
                            ...
                        ],
                     ),
                    ...
                ]
        allow_nonzero_column_sums : :obj:`bool`, default : False
            Traditionally, epidemiological models preserve the
            total population size. If that's not the case,
            switch off testing for this.
        reset_events : bool, default : True
            Whether to reset all linear events to zero before
            converting those.

        Example
        -------
        For an SEIR model with infection rate ``eta``.

        .. code:: python

            epi.set_quadratic_events([
                ( ("S", "I"),
                  eta,
                  [ ("S", -1), ("E", +1) ]
                ),
            ])

        Read  as

        "Coupling of *S* and *I* leads to
        the decay of one *S* particle to one *E* particle with
        rate :math:`\eta`.".
        """

        if reset_events:
            quadratic_event_updates = []
            quadratic_rate_functions = []
            quadratic_events = []
        else:
            quadratic_event_updates = list(self.quadratic_event_updates)
            quadratic_rate_functions = list(self.quadratic_rate_functions)
            quadratic_events = list(self.quadratic_events)

        for coupling_compartments, rate, affected_compartments in event_list:

            _s0 = self.get_compartment_id(coupling_compartments[0])
            _s1 = self.get_compartment_id(coupling_compartments[1])

            dy = np.zeros(self.N_comp)
            for trg, change in affected_compartments:
                _t = self.get_compartment_id(trg)
                dy[_t] += change

            if self._rate_has_functional_dependency(rate):
                this_rate = DynamicQuadraticRate(rate, _s0, _s1)
            else:
                this_rate = ConstantQuadraticRate(rate, _s0, _s1)

            quadratic_event_updates.append( dy )
            quadratic_rate_functions.append( this_rate )
            quadratic_events.append( (coupling_compartments, rate, affected_compartments) )

        if not allow_nonzero_column_sums and len(quadratic_rate_functions)>0:
            _y = np.ones(self.N_comp)
            test = sum([r(0,_y) * dy for dy, r in zip (quadratic_event_updates, quadratic_rate_functions)])
            test_sum = test.sum()
            if np.abs(test_sum) > 1e-15:
                warnings.warn("events do not sum to zero for each column:" + str(test_sum))

        self.quadratic_event_updates = quadratic_event_updates
        self.quadratic_rate_functions = quadratic_rate_functions
        self.quadratic_events = quadratic_events

        return self

    def dydt(self,t,y):
        """
        Compute the current momenta of the epidemiological model.

        Parameters
        ----------
        t : :obj:`float`
            Current time
        y : numpy.ndarray
            The entries correspond to the compartment frequencies
            (or counts, depending on population size).
        """
        
        ynew = sum([r(t,y) * dy for dy, r in zip(self.linear_event_updates, self.linear_rate_functions)])
        ynew += sum([r(t,y) * dy for dy, r in zip(self.birth_event_updates, self.birth_rate_functions)])
        if self.correct_for_dynamical_population_size:
            population_size = y.sum()
        else:
            population_size = self.initial_population_size
        ynew += sum([r(t,y)/population_size * dy for dy, r in zip(self.quadratic_event_updates, self.quadratic_rate_functions)])

        return ynew

    def get_numerical_dydt(self):
        """
        Return a function that obtains ``t`` and ``y`` as an input and returns ``dydt`` of this system
        """
        return self.dydt

    def get_time_leap_and_proposed_compartment_changes(self,
                                                       t,
                                                       current_event_rates = None, 
                                                       get_event_rates = None,
                                                       get_compartment_changes = None,
                                                       ):
        """
        For the current event rates, obtain a proposed
        time leap and concurrent state change vector.

        This method is needed for stochastic simulations.

        Parameters
        ----------
        t : float
            current time
        current_event_rates : list, default = None
            A list of constant rate values.
            Will be ignored if
            ``self.rates_have_explicit_time_dependence`` is ``True``,
            which is why ``None`` is a valid value.
        get_event_rates : function, default = None
            A function that takes time ``t`` and current
            state ``y`` as input and computes the rates of 
            all possible events.
            If ``None``, will attempt
            to set this to self.get_event_rates().
        get_compartment_changes : function, default = None
            A function that takes computed event rates
            and returns a random state change with
            probability proportional to its rate.
            If ``None``, will attempt
            to set this to self.get_compartment_changes().

        Returns
        -------
        tau : float
            A time leap.
        dy : numpy.ndarray
            A state change vector.
        """

        if get_event_rates is None:
            get_event_rates = self.get_event_rates
        if get_compartment_changes is None:
            get_compartment_changes = self.get_compartment_changes

        if self.rates_have_explicit_time_dependence:
            # solve the integral numerically
            if self.use_ivp_solver:
                new_t = time_leap_ivp(t, self.y0, get_event_rates)
            else:
                new_t = time_leap_newton(t, self.y0, get_event_rates)

            tau = new_t - t
            proposed_event_rates = get_event_rates(new_t, self.y0)
            dy = get_compartment_changes(proposed_event_rates)
        else:

            total_event_rate = current_event_rates.sum()
        
            tau = np.random.exponential(1/total_event_rate)
            dy = get_compartment_changes(current_event_rates)

        return tau, dy

    def get_compartment_changes(self, rates):
        """
        Sample a state change vector with probability
        proportional to its rate in ``rates``.
        
        Needed for stochastic simulations.

        Parameters
        ==========
        rates : numpy.ndarray
            A non-zero list of rates.
            Expects ``rates`` to be sorted according
            to 
            ``self.birth_event_updates + self.linear_event_updates + self.quadratic_event_updates``.

        Returns
        =======
        dy : numpy.ndarray
            A state change vector.
        """

        idy = custom_choice(rates/rates.sum())

        if idy < len(self.birth_event_updates):
            return self.birth_event_updates[idy]
        elif idy < len(self.birth_event_updates) + len(self.linear_event_updates):
            idy -= len(self.birth_event_updates)
            return self.linear_event_updates[idy]
        else:
            idy -= (len(self.birth_event_updates) + len(self.linear_event_updates))
            return self.quadratic_event_updates[idy]


    def get_event_rates(self, t, y):
        """
        Get a list of rate values corresponding to the previously
        set events.

        Parameters
        ----------
        t : float
            Current time
        y : numpy.ndarray
            Current state vector

        Returns
        -------
        rates : list
            A list of rate values corresponding to rates.
            Ordered as ``birth_rate_functions +
            linear_rate_functions + quadratic_rate_functions``.
        """
        rates = [r(t,y) for r in self.birth_rate_functions]
        rates += [r(t,y) for r in self.linear_rate_functions]
        if self.correct_for_dynamical_population_size:
            population_size = self.y0.sum()
        else:
            population_size = self.initial_population_size
        rates += [ r(t,self.y0)/population_size for r in self.quadratic_rate_functions ]
        rates = np.array(rates)

        return rates

    def get_numerical_event_and_rate_functions(self):
        """
        This function is needed to generalize
        stochastic simulations for child classes.

        Returns
        -------
        get_event_rates : func
            A function that takes the current time ``t`` and
            state vector ``y``
            and returns numerical event rate lists.
        get_compartment_changes : funx
            A function that takes a numerical list of event ``rates``
            and returns a random event state change vector
            with probability proportional to its entry in ``rates``.
        """
        return self.get_event_rates, self.get_compartment_changes

    def simulate(self,
                 tmax,
                 return_compartments=None,
                 sampling_dt=None,
                 sampling_callback=None,
                 adopt_final_state=False,
                 ):
        """
        Returns values of the given compartments at the demanded
        time points (as a numpy.ndarray of shape 
        ``(return_compartments), len(time_points)``.

        If ``return_compartments`` is None, all compartments will
        be returned.

        Parameters
        ----------
        tmax : float
            maximum length of the simulation
        return_compartments : list of compartments, default = None:
            The compartments for which to return time series.
            If ``None``, all compartments will be returned.
        sampling_dt : float, default = None
            Temporal distance between samples of the compartment counts.
            If ``None``, every change will be returned.
        sampling_callback : funtion, default = None
            A function that's called when a sample is taken

        Returns
        -------
        t : numpy.ndarray
            times at which compartment counts have been sampled
        result : dict
            Dictionary mapping a compartment to a time series of its count.
        """

        if return_compartments is None:
            return_compartments = self.compartments

        if sampling_callback is not None and sampling_dt is None:
            raise ValueError('A sampling callback function can only be set if sampling_dt is set, as well.')

        ndx = [self.get_compartment_id(C) for C in return_compartments]
        current_state = self.y0.copy()
        compartments = [ current_state.copy() ]

        if not adopt_final_state:
            initial_state = current_state.copy()
            initial_time = self.t0

        t = self.t0
        time = [self.t0]

        get_event_rates, get_compartment_changes = self.get_numerical_event_and_rate_functions()
        current_event_rates = get_event_rates(t, self.y0)
        total_event_rate = current_event_rates.sum()

        if sampling_callback is not None:
            sampling_callback()

        # Check for a) zero event rate and b) zero possibility for any nodes being changed still.
        # This is important because it might happen that nodes
        # have a non-zero reaction rate but no targets left
        # at which point the simulation will never halt.
        while t < tmax and \
              total_event_rate > 0:

            # sample and advance time according to current total rate
            tau, dy = self.get_time_leap_and_proposed_compartment_changes(t,
                                                                          current_event_rates=current_event_rates,
                                                                          get_event_rates=get_event_rates,
                                                                          get_compartment_changes=get_compartment_changes,
                                                                          )
            new_t = t + tau

            # break if simulation time is reached
            if new_t >= tmax:
                break

            # sampling
            if sampling_dt is not None:
                # sample all the time steps that were demanded in between the two events
                last_sample_dt = time[-1]
                for idt in range(1,int(np.ceil((new_t-last_sample_dt)/sampling_dt))):
                    time.append(last_sample_dt+idt*sampling_dt)
                    compartments.append(current_state.copy())
                    if sampling_callback is not None:
                        sampling_callback()

            # write losses and gains into the current state vector
            current_state += dy

            # save the current state if sampling_dt wasn't specified
            if sampling_dt is None:
                time.append(new_t)
                compartments.append(current_state.copy())

            # save current state
            self.t0 = new_t
            self.y0 = current_state.copy()

            current_event_rates = get_event_rates(new_t, self.y0)
            total_event_rate = current_event_rates.sum()

            # advance time
            t = new_t


        if sampling_dt is not None:
            next_sample = time[-1] + sampling_dt
            if next_sample <= tmax:
                time.append(next_sample)
                compartments.append(current_state)
                if sampling_callback is not None:
                    sampling_callback()

        # convert to result dictionary
        time = np.array(time)
        result = np.array(compartments)

        if not adopt_final_state:
            self.y0 = initial_state
            self.t0 = initial_time
        else:
            self.t0 = tmax


        return time, { compartment: result[:,c_ndx] for c_ndx, compartment in zip(ndx, return_compartments) }

class SIModel(EpiModel):
    """
    An SI model derived from :class:`epipack.numeric_epi_models.EpiModel`.
    """

    def __init__(self, infection_rate, initial_population_size=1.0):

        EpiModel.__init__(self, list("SI"), initial_population_size)

        self.set_processes([
                ("S", "I", infection_rate, "I", "I"),
            ])


class SISModel(EpiModel):
    """
    An SIS model derived from :class:`epipack.numeric_epi_models.EpiModel`.

    Parameters
    ----------
    R0 : float
        The basic reproduction number. From this, the infection
        rate is computed as ``infection_rate = R0 * recovery_rate``
    recovery_rate : float
        Recovery rate
    population_size : float, default = 1.0
        Number of people in the population.
    """

    def __init__(self, infection_rate, recovery_rate, initial_population_size=1.0):

        EpiModel.__init__(self, list("SI"), initial_population_size)

        self.set_processes([
                ("S", "I", infection_rate, "I", "I"),
                ("I", recovery_rate, "S" ),
            ])
class SIRModel(EpiModel):
    """
    An SIR model derived from :class:`epipack.numeric_epi_models.EpiModel`.
    """

    def __init__(self, infection_rate, recovery_rate, initial_population_size=1.0):

        EpiModel.__init__(self, list("SIR"), initial_population_size)

        self.set_processes([
                ("S", "I", infection_rate, "I", "I"),
                ("I", recovery_rate, "R"),
            ])

class SIRSModel(EpiModel):
    """
    An SIRS model derived from :class:`epipack.numeric_epi_models.EpiModel`.
    """

    def __init__(self, infection_rate, recovery_rate, waning_immunity_rate, initial_population_size=1.0):

        EpiModel.__init__(self, list("SIR"), initial_population_size)

        self.set_processes([
                ("S", "I", infection_rate, "I", "I"),
                ("I", recovery_rate, "R"),
                ("R", waning_immunity_rate, "S"),
            ])

class SEIRModel(EpiModel):
    """
    An SEIR model derived from :class:`epipack.numeric_epi_models.EpiModel`.
    """

    def __init__(self, infection_rate, recovery_rate, symptomatic_rate, initial_population_size=1.0):

        EpiModel.__init__(self, list("SEIR"), initial_population_size)

        self.set_processes([
                ("S", "I", infection_rate, "E", "I"),
                ("E", symptomatic_rate, "I"),
                ("I", recovery_rate, "R"),
            ])

if __name__=="__main__":    # pragma: no cover
    N = 100
    epi = EpiModel(list("SEIR"),100)
    #print(epi.compartments)
    #print()
    epi.set_processes([
            ("S", "I", 2.0, "E", "I"),
            ("E", 1.0, "I"),
            ("I", 1.0, "R"),
            ])

    print("#printing updates")
    #print([dy.toarray() for dy in epi.linear_event_updates])
    #print([dy.toarray() for dy in epi.quadratic_event_updates])

    import matplotlib.pyplot as pl
    from time import time

    N_meas = 5

    tt = np.linspace(0,20,100)
    start = time()
    epi = EpiModel(list("SEIR"),100)
    epi.set_processes([
                ("S", "I", 2.0, "E", "I"),
                ("E", 1.0, "I"),
                ("I", 1.0, "R"),
            ])
    for meas in range(N_meas):
        epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
        result = epi.integrate(tt)
    end = time()

    print("arrays needed", end-start,"s")

    pl.plot(tt, result['S'],label='S')
    pl.plot(tt, result['E'],label='E')
    pl.plot(tt, result['I'],label='I')
    pl.plot(tt, result['R'],label='R')
    pl.legend()

    from epipack import MatrixSEIRModel

    tt = np.linspace(0,20,50)
    SEIR = MatrixSEIRModel(2.0,1.0,1.0,initial_population_size=N)
    SEIR.set_initial_conditions({'S':0.99*N,'I':0.01*N})
    result = SEIR.integrate(tt)
    pl.plot(tt, result['S'],'s',label='S',mfc='None')
    pl.plot(tt, result['E'],'s',label='E',mfc='None')
    pl.plot(tt, result['I'],'s',label='I',mfc='None')
    pl.plot(tt, result['R'],'s',label='R',mfc='None')



    ##########
    epi = EpiModel(list("SEIR"),100)
    epi.set_processes([
                ("S", "I", 2.0, "E", "I"),
                ("E", 1.0, "I"),
                ("I", 1.0, "R"),
            ])
    epi.set_initial_conditions({'S':99,'I':1})
    t, result = epi.simulate(tt[-1])
    pl.plot(t, result['S'],label='S')
    pl.plot(t, result['E'],label='E')
    pl.plot(t, result['I'],label='I')
    pl.plot(t, result['R'],label='R')

    pl.figure()

    S, I, R = list("SIR")
    N = 200
    model = EpiModel([S,I],N,correct_for_dynamical_population_size=True)

    def temporalR0(t,y):
        return 4 + np.cos(t/100*2*np.pi)

    model.set_processes([
            (S, I, temporalR0, I, I),
            (None, N, S),
            (I, 1, S),
            (S, 1, None),
            (I, 1, None),
        ])
    model.set_initial_conditions({
            S: 190,
            I: 10,
        })

    def print_status():
        print(model.t0/150*100)
    t, result = model.simulate(150,sampling_dt=0.5,sampling_callback=print_status)
    pl.plot(t, result['S'],label='S')
    pl.plot(t, result['I'],label='I')

    model.set_initial_conditions({
            S: 190,
            I: 10,
        })
    tt = np.linspace(0,100,1000)
    result = model.integrate(tt)
    pl.plot(tt, result['S'],label='S')
    pl.plot(tt, result['I'],label='I')

    pl.show()

