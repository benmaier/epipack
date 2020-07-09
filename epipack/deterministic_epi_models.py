"""
Provides an API to define deterministic epidemiological models.
"""

import numpy as np 
import scipy.sparse as sprs

import warnings

from epipack.integrators import integrate_dopri5
from epipack.process_conversions import (
            processes_to_rates,
            transition_processes_to_rates,
            fission_processes_to_rates,
            fusion_processes_to_rates,
            transmission_processes_to_rates,
        )

class DeterministicEpiModel():
    """
    A general class to define standard 
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
    linear_rates : scipy.sparse.csr_matrix
        Sparse matrix containing 
        transition rates of the linear processes.
    quadratic_rates : scipy.sparse.csr_matrix
        List of sparse matrices that contain
        transition rates of the quadratic processes
        for each compartment.
    affected_by_quadratic_process : :obj:`list` of :obj:`int`
        List of integer compartment IDs, collecting
        compartments that are affected
        by the quadratic processes


    Example
    -------

    .. code:: python
        
        >>> epi = DeterministicEpiModel(["S","I","R"])
        >>> print(epi.compartments)
        [ "S", "I", "R" ]


    """

    def __init__(self,compartments,population_size=1):
        """
        """

        self.y0 = None
        self.affected_by_quadratic_process = None

        self.compartments = list(compartments)
        self.population_size = population_size
        self.N_comp = len(self.compartments)
        self.birth_rates = np.zeros((self.N_comp,),dtype=np.float64)
        self.linear_rates = sprs.csr_matrix((self.N_comp, self.N_comp),dtype=np.float64) 
        self.quadratic_rates = [ sprs.csr_matrix((self.N_comp, self.N_comp),dtype=np.float64)\
                                 for c in range(self.N_comp) ]

    def get_compartment_id(self,C):
        """Get the integer ID of a compartment ``C``"""
        return self.compartments.index(C)

    def get_compartment(self,iC):
        """Get the compartment, given an integer ID ``iC``"""
        return self.compartments[iC]

    def set_processes(self,process_list,allow_nonzero_column_sums=False,reset_rates=True,
                      ignore_rate_position_checks=False):
        """
        Converts a list of reaction process tuples to rate tuples and sets the rates for this model.

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
        reset_rates : bool, default : True
            If this is `True`, reset all rates to zero before setting the new ones.
        ignore_rate_position_checks : bool, default = False
            This function usually checks whether the rate of 
            a reaction is positioned correctly. You can
            turn this behavior off for transition, birth, death, and
            transmission processes. (Useful if you want to define
            symbolic transmission processes that are compartment-dependent).


        """

        quadratic_rates, linear_rates = processes_to_rates(process_list, self.compartments,ignore_rate_position_checks)
        self.set_linear_rates(linear_rates,allow_nonzero_column_sums=allow_nonzero_column_sums)
        self.set_quadratic_rates(quadratic_rates,allow_nonzero_column_sums=allow_nonzero_column_sums)

        return self

    def set_linear_rates(self,rate_list,allow_nonzero_column_sums=False,reset_rates=True):
        """
        Define the linear transition rates between compartments.

        Parameters
        ==========

        rate_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions rates in the following format:

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

        reset_rates : bool, default : True
            Whether to reset all linear rates to zero before 
            converting those.
        """

        if reset_rates:
            linear_rates = sprs.lil_matrix((self.N_comp, self.N_comp),dtype=np.float64) 
            birth_rates = np.zeros((self.N_comp,),dtype=np.float64)
        else:
            linear_rates = self.linear_rates.tolil()
            birth_rates = self.birth_rates.copy()

        for acting_compartment, affected_compartment, rate in rate_list:

            _t = self.get_compartment_id(affected_compartment)
            if acting_compartment is None:
                birth_rates[_t] += rate
            else:
                _s = self.get_compartment_id(acting_compartment)
                linear_rates[_t, _s] += rate


        linear_rates = linear_rates.tocsr()

        if not allow_nonzero_column_sums:
            test = linear_rates.sum(axis=0) + birth_rates
            test_sum = test.sum()
            if np.abs(test_sum) > 1e-15:
                warnings.warn("Rates do not sum to zero for each column:" + str(test_sum))

        self.linear_rates = linear_rates
        self.birth_rates = birth_rates

        return self


    def add_transition_processes(self,process_list):
        """
        Define the linear transition processes between compartments.

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

            epi.add_transition_processes([
                ("E", symptomatic_rate, "I" ),
                ("I", recovery_rate, "R" ),
            ])

        """

        linear_rates = transition_processes_to_rates(process_list)

        return self.set_linear_rates(linear_rates, reset_rates=False, allow_nonzero_column_sums=True)

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
                ("B", growth_rate, "B", "B" ),
            ])

        """
        linear_rates = fission_processes_to_rates(process_list)

        return self.set_linear_rates(linear_rates, reset_rates=False, allow_nonzero_column_sums=True)
    
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
        quadratic_rates = fusion_processes_to_rates(process_list)

        return self.set_quadratic_rates(quadratic_rates, reset_rates=False, allow_nonzero_column_sums=True)

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
        quadratic_rates = transmission_processes_to_rates(process_list)

        return self.set_quadratic_rates(quadratic_rates, reset_rates=False, allow_nonzero_column_sums=True)

    def add_quadratic_rates(self,rate_list,reset_rates=True,allow_nonzero_column_sums=False):
        """
        Add quadratic rates without resetting the existing rate terms.
        See :func:`_tacoma.set_quadratic_rates` for docstring.
        """

        return self.set_quadratic_rates(rate_list,reset_rates=False,allow_nonzero_column_sums=allow_nonzero_column_sums)

    def add_linear_rates(self,rate_list,reset_rates=True,allow_nonzero_column_sums=False):
        """
        Add linear rates without resetting the existing rate terms.
        See :func:`_tacoma.set_linear_rates` for docstring.
        """

        return self.set_linear_rates(rate_list,reset_rates=False,allow_nonzero_column_sums=allow_nonzero_column_sums)
    
    def set_quadratic_rates(self,rate_list,reset_rates=True,allow_nonzero_column_sums=False):
        r"""
        Define the quadratic transition processes between compartments.

        Parameters
        ----------
        rate_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions rates in the following format:

            .. code:: python

                [
                    ("coupling_compartment_0", 
                     "coupling_compartment_1",
                     "affected_compartment", 
                     rate 
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

            epi.set_quadratic_rates([
                ("S", "I", "S", -1 ),
                ("S", "I", "E", +1 )
            ])

        Read  as
        
        "Coupling of *S* and *I* leads to 
        a reduction in "S" proportional to :math:`S\times I` and rate -1/time_unit.
        Furthermore, coupling of *S* and *I* leads to 
        an increase in "E" proportional to :math:`S\times I` and rate +1/time_unit."
        """

        if reset_rates:

            matrices = [None for c in self.compartments]
            for c in range(self.N_comp):
                matrices[c] = sprs.lil_matrix(
                                    (self.N_comp, self.N_comp),
                                    dtype=np.float64,
                                   )

            all_affected = []
        else:
            matrices =  [ M.tolil() for M in self.quadratic_rates ]
            all_affected = self.affected_by_quadratic_process if self.affected_by_quadratic_process is not None else []
        
        for coupling0, coupling1, affected, rate in rate_list:

            c0, c1 = sorted([ self.get_compartment_id(c) for c in [coupling0, coupling1] ])
            a = self.get_compartment_id(affected)

            matrices[a][c0,c1] += rate
            all_affected.append(a)

        total_sum = 0
        for c in range(self.N_comp):
            matrices[c] = matrices[c].tocsr()
            total_sum += matrices[c].sum(axis=0).sum()

        if np.abs(total_sum) > 1e-14:
            if not allow_nonzero_column_sums:
                warnings.warn("Rates do not sum to zero. Sum = "+ str(total_sum))

        self.affected_by_quadratic_process = sorted(list(set(all_affected)))
        self.quadratic_rates = matrices

        return self

    def dydt(self,t,y,quadratic_global_factor=None):
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
        quadratic_global_factor : :obj:`func`, default : None
            A function that takes the current state as an
            input and returns a global factor with which
            the quadratic rates will be modulated.
        """
        
        ynew = self.linear_rates.dot(y) + self.birth_rates
        f = quadratic_global_factor(y) if quadratic_global_factor is not None else 1.0
        for c in self.affected_by_quadratic_process:
            ynew[c] += f*y.T.dot(self.quadratic_rates[c].dot(y)) / self.population_size
            
        return ynew



    def set_initial_conditions(self, initial_conditions,allow_nonzero_column_sums=False):
        """
        Set the initial conditions for integration

        Parameters
        ----------
        initial_conditions : dict
            A dictionary that maps compartments to a fraction
            of the population. Compartments that are not
            set in this dictionary are assumed to have an initial condition
            of zero.
        allow_nonzero_column_sums : bool, default = False
            If True, an error is raised when the initial conditions do not
            sum to the population size.
        """

        if type(initial_conditions) == dict:
            initial_conditions = list(initial_conditions.items())

        self.y0 = np.zeros(self.N_comp)
        total = 0
        for compartment, amount in initial_conditions:
            total += amount
            if self.y0[self.get_compartment_id(compartment)] != 0:
                warnings.warn('Double entry in initial conditions for compartment '+str(compartment))
            else:
                self.y0[self.get_compartment_id(compartment)] = amount

        if np.abs(total-self.population_size)/self.population_size > 1e-14 and not allow_nonzero_column_sums:
            warnings.warn('Sum of initial conditions does not equal unity.')

        return self

    def integrate_and_return_by_index(self,time_points,return_compartments=None):
        """
        Returns values of the given compartments at the demanded
        time points (as a numpy.ndarray of shape 
        ``(return_compartments), len(time_points)``.

        If ``return_compartments`` is None, all compartments will
        be returned.
        """

        if return_compartments is None:
            return_compartments = self.compartments
        result = integrate_dopri5(self.dydt, time_points, self.y0)

        ndx = [self.get_compartment_id(C) for C in return_compartments]

        return result[ndx,:]

    def integrate(self,time_points,return_compartments=None):
        """
        Returns values of the given compartments at the demanded
        time points (as a dictionary).

        If ``return_compartments`` is None, all compartments will
        be returned.
        """
        if return_compartments is None:
            return_compartments = self.compartments
        result = self.integrate_and_return_by_index(time_points, return_compartments)

        result_dict = {}
        for icomp, compartment in enumerate(return_compartments):
            result_dict[compartment] = result[icomp,:]

        return result_dict


class DeterministicSIModel(DeterministicEpiModel):
    """
    An SI model derived from :class:`epipack.deterministic_epi_models.DeterministicEpiModel`.
    """

    def __init__(self, infection_rate, population_size=1.0):

        DeterministicEpiModel.__init__(self, list("SI"), population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])


class DeterministicSISModel(DeterministicEpiModel):
    """
    An SIS model derived from :class:`epipack.deterministic_epi_models.DeterministicEpiModel`.

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

    def __init__(self, R0, recovery_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SI"), population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])
        self.add_transition_processes([
                ("I", recovery_rate, "S" ),
            ])

class DeterministicSIRModel(DeterministicEpiModel):
    """
    An SIR model derived from :class:`epipack.deterministic_epi_models.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SIR"), population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])
        self.add_transition_processes([
                ("I", recovery_rate, "R"),
            ])

class DeterministicSIRXModel(DeterministicEpiModel):
    """
    An SIRX model derived from :class:`epipack.deterministic_epi_models.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, quarantine_rate, containment_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SIRXH"), population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])
        self.add_transition_processes([
                ("S", containment_rate, "H"),
                ("I", recovery_rate, "R"),
                ("I", containment_rate+quarantine_rate, "X"),
            ])

class DeterministicSEIRXModel(DeterministicEpiModel):
    """
    An SEIRX model derived from :class:`epipack.deterministic_epi_models.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, symptomatic_rate, quarantine_rate, containment_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SEIRXH"), population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "E", +infection_rate),
            ])
        self.add_transition_processes([
                ("E", symptomatic_rate                ,"I"),
                ("S", containment_rate               ,"H"), 
                ("I", recovery_rate                  ,"R"),
                ("I", containment_rate+quarantine_rate,"X")
            ])

class DeterministicSIRSModel(DeterministicEpiModel):
    """
    An SIRS model derived from :class:`epipack.deterministic_epi_models.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, waning_immunity_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SIR"), population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])
        self.add_transition_processes([
                ("I", recovery_rate, "R"),
                ("R", waning_immunity_rate, "S"),
            ])

class DeterministicSEIRModel(DeterministicEpiModel):
    """
    An SEIR model derived from :class:`epipack.deterministic_epi_models.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, symptomatic_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SEIR"), population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "E", +infection_rate),
            ])
        self.add_transition_processes([
                ("E", symptomatic_rate, "I"),
                ("I", recovery_rate, "R"),
            ])

class DeterministicSEIRSModel(DeterministicEpiModel):
    """
    An SEIRS model derived from :class:`epipack.deterministic_epi_models.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, symptomatic_rate, waning_immunity_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SEIR"), population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "E", +infection_rate),
            ])
        self.add_transition_processes([
                ("E", symptomatic_rate      , "I"),
                ("I", recovery_rate         , "R"),
                ("R", waning_immunity_rate , "S"),
            ])


if __name__=="__main__":
    epi = DeterministicEpiModel(list("SEIR"))
    print(epi.compartments)
    print()
    epi.add_transition_processes([
            ("E", 1.0, "I"),
            ("I", 1.0, "R"),
            ])
    print(epi.linear_rates)
    epi.set_quadratic_rates([
            ("S", "I", "S", -1.0),
            ("S", "I", "E", +1.0),
            ])
    print()
    for iM, M in enumerate(epi.quadratic_rates):
        print(epi.get_compartment(iM), M)

    import matplotlib.pyplot as pl

    N = 100
    epi = DeterministicSISModel(R0=2,recovery_rate=1,population_size=N)
    print(epi.linear_rates)
    epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
    tt = np.linspace(0,10,100)
    result = epi.integrate(tt,['S','I'])

    pl.plot(tt, result['S'],label='S')
    pl.plot(tt, result['I'],label='I')
    pl.legend()

    pl.show()
