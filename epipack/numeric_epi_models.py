"""
Provides an API to define Numeric epidemiological models.
"""

import numpy as np 
import scipy.sparse as sprs

import warnings

from epipack.integrators import (
            integrate_dopri5,
            integrate_euler,
            IntegrationMixin,
        )

from epipack.stochastic_epi_models import (
            SimulationMixin,
        )

from epipack.process_conversions import (
            processes_to_events,
            transition_processes_to_events,
            fission_processes_to_events,
            fusion_processes_to_events,
            transmission_processes_to_events,
        )

class NumericEpiModel(IntegrationMixin,SimulationMixin):    
    """
    A general class to define a standard 
    mean-field compartmental
    epidemiological model.

    Parameters
    ----------

    compartments : :obj:`list` of :obj:`string`
        A list containing compartment strings.

    Attributes
    ----------

    compartments : :obj:`list` of :obj:`string`
        A list containing strings that describe each compartment,
        (e.g. "S", "I", etc.).
    N_comp : :obj:`int`
        Number of compartments (including population number)
    linear_events : scipy.sparse.csr_matrix
        Sparse matrix containing 
        transition events of the linear processes.
    quadratic_events : scipy.sparse.csr_matrix
        List of sparse matrices that contain
        transition events of the quadratic processes
        for each compartment.
    affected_by_quadratic_process : :obj:`list` of :obj:`int`
        List of integer compartment IDs, collecting
        compartments that are affected
        by the quadratic processes


    Example
    -------

    .. code:: python
        
        >>> epi = NumericEpiModel(["S","I","R"])
        >>> print(epi.compartments)
        [ "S", "I", "R" ]


    """

    def __init__(self,compartments,initial_population_size=1,correct_for_dynamical_population_size=False):
        """
        """

        self.y0 = None

        self.compartments = list(compartments)
        self.compartment_ids = { C: iC for iC, C in enumerate(compartments) }
        self.initial_population_size = initial_population_size
        self.correct_for_dynamical_population_size = correct_for_dynamical_population_size
        self.N_comp = len(self.compartments)
        self.birth_rate_functions = []
        self.birth_event_updates = []
        self.linear_rate_functions = []
        self.linear_event_updates = []
        self.quadratic_rate_functions = []
        self.quadratic_event_updates = []

    def get_compartment_id(self,C):
        """Get the integer ID of a compartment ``C``"""
        return self.compartment_ids[C]

    def get_compartment(self,iC):
        """Get the compartment, given an integer ID ``iC``"""
        return self.compartments[iC]

    def set_processes(self,process_list,allow_nonzero_column_sums=False,reset_events=True,
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

        quadratic_events, linear_events = processes_to_events(process_list, self.compartments,ignore_rate_position_checks)
        self.set_linear_events(linear_events,allow_nonzero_column_sums=allow_nonzero_column_sums)
        self.set_quadratic_events(quadratic_events,allow_nonzero_column_sums=allow_nonzero_column_sums)

        return self

    def _rate_has_functional_dependency(self,rate):
        return callable(rate)

    def set_linear_events(self,event_list,allow_nonzero_column_sums=False,reset_events=True):
        """
        Define the linear transition events between compartments.

        Parameters
        ==========

        event_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions events in the following format:

            .. code:: python

                [
                    ( acting_compartment, affected_compartment, rate ),
                    ...
                ]

        allow_nonzero_column_sums : :obj:`bool`, default : False
            Traditionally, epidemiological models preserve the
            total population size. If that's not the case,
            switch off testing for this.

        Example

        reset_events : bool, default : True
            Whether to reset all linear events to zero before 
            converting those.
        """

        if reset_events:
            birth_rate_functions = []
            birth_event_updates = []
            linear_rate_functions = []
            linear_event_updates = []
        else:
            linear_events = self.linear_event_updates
            birth_events = self.birth_event_updates

        for acting_compartments, rate, affected_compartments in event_list:

            data, row, col = [], [], []
            for trg, change in affected_compartments:
                col.append(self.get_compartment_id(trg))
                data.append(change)
                row.append(0)
            dy =  sprs.coo_matrix((data,(row,col)),shape=(1,self.N_comp),dtype=float).tocsr()

            if acting_compartments[0] is None:
                if self._rate_has_functional_dependency(rate):
                    this_rate = rate
                else:
                    this_rate = lambda t, y: rate
                birth_event_updates.append( dy )
                birth_rate_functions.append( this_rate )
            else:
                _s = self.get_compartment_id(acting_compartments[0])
                if self._rate_has_functional_dependency(rate):
                    this_rate = lambda t, y: rate(t,y)*y[_s]
                else:
                    this_rate = lambda t, y: rate*y[_s]
                linear_event_updates.append( dy )
                linear_rate_functions.append( this_rate )


        if not allow_nonzero_column_sums:
            _y = np.ones(self.N_comp)
            test = sum([r(0,_y) * dy for dy, r in zip (linear_event_updates, linear_rate_functions)])
            test += sum([r(0,_y) * dy for dy, r in zip (birth_event_updates, birth_rate_functions)])
            test_sum = test.toarray().flatten().sum()
            if np.abs(test_sum) > 1e-15:
                warnings.warn("events do not sum to zero for each column:" + str(test_sum))

        self.linear_event_updates = linear_event_updates
        self.birth_rate_functions = birth_rate_functions

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
        -------

        For an SEIR model.

        .. code:: python

            epi.add_transition_processes([
                ("E", symptomatic_rate, "I" ),
                ("I", recovery_rate, "R" ),
            ])

        """

        linear_events = transition_processes_to_events(process_list)

        return self.set_linear_events(linear_events, reset_events=False, allow_nonzero_column_sums=True)

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

        return self.set_linear_events(linear_events, reset_events=False, allow_nonzero_column_sums=True)
    
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

        return self.set_quadratic_events(quadratic_events, reset_events=False, allow_nonzero_column_sums=True)

    def add_transmission_processes(self,process_list):
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

            epi.add_transmission_processes([
                ("I", "S", +1, "I", "E" ),
            ])

        """
        quadratic_events = transmission_processes_to_events(process_list)

        return self.set_quadratic_events(quadratic_events, reset_events=False, allow_nonzero_column_sums=True)

    def add_quadratic_events(self,event_list,reset_events=True,allow_nonzero_column_sums=False):
        """
        Add quadratic events without resetting the existing event terms.
        See :func:`_tacoma.set_quadratic_events` for docstring.
        """

        return self.set_quadratic_events(event_list,reset_events=False,allow_nonzero_column_sums=allow_nonzero_column_sums)

    def add_linear_events(self,event_list,reset_events=True,allow_nonzero_column_sums=False):
        """
        Add linear events without resetting the existing event terms.
        See :func:`_tacoma.set_linear_events` for docstring.
        """

        return self.set_linear_events(event_list,reset_events=False,allow_nonzero_column_sums=allow_nonzero_column_sums)
    
    def set_quadratic_events(self,event_list,reset_events=True,allow_nonzero_column_sums=False):
        r"""
        Define the quadratic transition processes between compartments.

        Parameters
        ----------
        event_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions events in the following format:

            .. code:: python

                [
                    ("coupling_compartment_0", 
                     "coupling_compartment_1",
                     "affected_compartment", 
                     event 
                     ),
                    ...
                ]

        allow_nonzero_column_sums : :obj:`bool`, default : False
            Traditionally, epidemiological models preserve the
            total population size. If that's not the case,
            switch off testing for this.

        Example
        -------

        For an SEIR model.

        .. code:: python

            epi.set_quadratic_events([
                ("S", "I", "S", -1 ),
                ("S", "I", "E", +1 )
            ])

        Read  as
        
        "Coupling of *S* and *I* leads to 
        a reduction in "S" proportional to :math:`S\times I` and event -1/time_unit.
        Furthermore, coupling of *S* and *I* leads to 
        an increase in "E" proportional to :math:`S\times I` and event +1/time_unit."
        """

        if reset_events:
            quadratic_event_updates = []
            quadratic_rate_functions = []
        else:
            quadratic_event_updates = self.quadratic_event_updates
            quadratic_rate_functions = self.quadratic_rate_functions
        
        for coupling_compartments, rate, affected_compartments in event_list:

            _s0 = self.get_compartment_id(coupling_compartments[0])
            _s1 = self.get_compartment_id(coupling_compartments[1])

            data, row, col = [], [], []
            for trg, change in affected_compartments:
                col.append(self.get_compartment_id(trg))
                data.append(change)
                row.append(0)
            dy =  sprs.coo_matrix((data,(row,col)),shape=(1,self.N_comp),dtype=float).tocsr()

            if self._rate_has_functional_dependency(rate):
                this_rate = lambda t, y: rate(t,y) * y[_s0] * y[_s1]
            else:
                this_rate = lambda t, y: rate * y[_s0] * y[_s1]

            quadratic_event_updates.append( dy )
            quadratic_rate_functions.append( this_rate )


        if not allow_nonzero_column_sums:
            _y = np.ones(self.N_comp)
            test = sum([r(0,_y) * dy for dy, r in zip (quadratic_event_updates, quadratic_rate_functions)])
            test_sum = test.toarray().flatten().sum()
            if np.abs(test_sum) > 1e-15:
                warnings.warn("events do not sum to zero for each column:" + str(test_sum))

        self.quadratic_event_updates = quadratic_event_updates
        self.quadratic_rate_functions = quadratic_rate_functions

        return self

    def dydt(self,t,y):
        """
        Compute the current momenta of the epidemiological model.

        Parameters
        ----------
        t : :obj:`float`
            Current time
        y : numpy.ndarray
            The first entry is equal to the population size.
            The remaining entries are equal to the current fractions
            of the population of the respective compartments
        """
        
        ynew = sum([r(t,y) * dy for dy, r in zip (self.linear_event_updates, self.linear_rate_functions)])
        ynew += sum([r(t,y) * dy for dy, r in zip (self.birth_event_updates, self.birth_rate_functions)])
        if self.correct_for_dynamical_population_size:
            population_size = y.sum()
        else:
            population_size = self.initial_population_size
        ynew += sum([r(t,y)/population_size * dy for dy, r in zip (self.quadratic_event_updates, self.quadratic_rate_functions)])
            
        return ynew.toarray().flatten()

    def get_numerical_dydt(self):
        """
        Return a function that obtains t and y as an input and returns dydt of this system
        """
        return self.dydt



if __name__=="__main__":    # pragma: no cover
    N = 100
    epi = NumericEpiModel(list("SEIR"),100)
    print(epi.compartments)
    print()
    epi.set_processes([
            ("E", 1.0, "I"),
            ("I", 1.0, "R"),
            ("S", "I", 2.0, "E", "I"),
            ])

    import matplotlib.pyplot as pl

    epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
    tt = np.linspace(0,1000,100)
    result = epi.integrate(tt)

    pl.plot(tt, result['S'],label='S')
    pl.plot(tt, result['I'],label='I')
    pl.plot(tt, result['E'],label='E')
    pl.plot(tt, result['R'],label='R')
    pl.legend()

    pl.show()
