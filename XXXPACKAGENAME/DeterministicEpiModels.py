"""
Provides an API to define epidemiological models.
"""

import numpy as np 
import scipy.sparse as sprs

from epidemicmodeling.integrators import integrate_dopri5

class DeterministicEpiModel():
    """
    A general class to define any 
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
    mobility : scipy.sparse.csr_matrix
        Sparse matrix containing whether a compartment
        is mobile or not.
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
        self.mobility = sprs.eye(self.N_comp,format='csr') 
        self.linear_rates = sprs.csr_matrix((self.N_comp, self.N_comp),dtype=np.float64) 
        self.quadratic_rates = [ sprs.csr_matrix((self.N_comp, self.N_comp),dtype=np.float64)\
                                 for c in range(self.N_comp) ]
        # TODO: Make a version where the last compartment is ignored and always given by 1 - sum over remaining

    def get_compartment_id(self,C):
        """Get the integer ID of a compartment ``C``"""
        return self.compartments.index(C)

    def get_compartment(self,iC):
        """Get the compartment, given an integer ID ``iC``"""
        return self.compartments[iC]

    def set_compartment_mobility(self,mobility):
        """
        Define whether compartments are mobile. Note
        that as per default, all compartments will be considered
        to be mobile, so in principle you only have to define the
        compartments that are immobile.

        Parameters
        ==========

        mobility : :obj:`list` of :obj:`tuple`
            A dictionary that contains Boolean information about compartment
            mobility

            .. code:: python

                {
                    "compartment0": True ,  # if mobile
                    "compartment1": False , # if not mobile
                    ...
                }

        """
        
        # initiate with population number N, which is always mobile
        data = []
        row_ind = []
        col_ind = []

        this_mobility = sprs.eye(self.N_comp, dtype=np.float64).tolil()

        not_given = []
        for C, mob in mobility.items():
            if C not in self.compartments:
                not_given.append(C)
        if len(not_given) > 0:
            raise ValueError("The following compartments are undefined: ", not_given)

        for compartment, is_mobile in mobility.items():
            c = self.get_compartment_id(compartment)

            if is_mobile:
                this_mobility[c,c] = 1.0
            else:
                this_mobility[c,c] = 0.0
            row_ind.append(c)
            col_ind.append(c)

        self.mobility = this_mobility.tocsr()

        return self
        

    def set_linear_processes(self,rate_list,allow_nonzero_column_sums=False):
        """
        Define the linear transition processes between compartments.

        Parameters
        ==========

        rate_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions rates in the following format:

            .. code:: python

                [
                    ("source_compartment", "target_compartment", rate ),
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

            epi.set_linear_processes([
                ("E", "I", symptomatic_rate ),
                ("I", "R", recovery_rate ),
            ])

        """

        linear_rates = sprs.lil_matrix(
                                (self.N_comp, self.N_comp),
                                dtype=np.float64,
                               )
        
        for source, target, rate in rate_list:

            _s = self.get_compartment_id(source)
            _t = self.get_compartment_id(target)
            
            # source compartment loses an entity
            # target compartment gains one
            linear_rates[_s, _s] += -rate
            linear_rates[_t, _s] += rate

        linear_rates = linear_rates.tocsr()

        if not allow_nonzero_column_sums:
            test = linear_rates.sum(axis=0)
            test_sum = test.sum()
            if np.abs(test_sum) > 1e-15:
                raise ValueError("Rates do not sum to zero for each column:" + str(test_sum))

        self.linear_rates = linear_rates

        return self

    def set_quadratic_processes(self,rate_list,allow_nonzero_column_sums=False):
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

            epi.set_quadratic_processes([
                ("S", "I", "S", -1 ),
                ("S", "I", "E", +1 )
            ])

        Read  as
        
        "Coupling of *S* and *I* leads to 
        a reduction in "S" proportional to :math:`S\times I` and rate -1/time_unit.
        Furthermore, coupling of *S* and *I* leads to 
        an increase in "E" proportional to :math:`S\times I` and rate +1/time_unit."
        """

        matrices = [None for c in self.compartments]
        for c in range(self.N_comp):
            matrices[c] = sprs.lil_matrix(
                                (self.N_comp, self.N_comp),
                                dtype=np.float64,
                               )

        all_affected = []
        
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
                raise ValueError("Rates do not sum to zero. Sum = "+ str(total_sum))

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
        
        ynew = self.linear_rates.dot(y)
        f = quadratic_global_factor(y) if quadratic_global_factor is not None else 1.0
        for c in self.affected_by_quadratic_process:
            ynew[c] += f*y.dot(self.quadratic_rates[c].dot(y)) / self.population_size
            
        return ynew

    def set_initial_conditions(self, initial_conditions):
        """
        """

        if type(initial_conditions) == dict:
            initial_conditions = list(initial_conditions.items())

        self.y0 = np.zeros(self.N_comp)
        total = 0
        for compartment, amount in initial_conditions:
            total += amount
            if self.y0[self.get_compartment_id(compartment)] != 0:
                raise ValueError('Double entry in initial conditions for compartment '+str(compartment))
            else:
                self.y0[self.get_compartment_id(compartment)] = amount

        if np.abs(total-self.population_size)/self.population_size > 1e-14:
            raise ValueError('Sum of initial conditions does not equal unity.')        

        return self

    def get_result(self,time_points,return_compartments=None):
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
        result = self.get_result(time_points, return_compartments)

        result_dict = {}
        for icomp, compartment in enumerate(return_compartments):
            result_dict[compartment] = result[icomp,:]

        return result_dict






class SIModel(DeterministicEpiModel):
    """
    An SI model derived from :class:`metapop.epi.DeterministicEpiModel`.
    """

    def __init__(self, infection_rate, population_size=1.0):

        DeterministicEpiModel.__init__(self, list("SI"), population_size)

        self.set_quadratic_processes([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])


class SISModel(DeterministicEpiModel):
    """
    An SIS model derived from :class:`metapop.epi.DeterministicEpiModel`.

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

        self.set_quadratic_processes([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])
        self.set_linear_processes([
                ("I", "S", recovery_rate),
            ])

class SIRModel(DeterministicEpiModel):
    """
    An SIR model derived from :class:`metapop.epi.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SIR"), population_size)

        self.set_quadratic_processes([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])
        self.set_linear_processes([
                ("I", "R", recovery_rate),
            ])

class SIRXModel(DeterministicEpiModel):
    """
    An SIRX model derived from :class:`metapop.epi.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, quarantine_rate, containment_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SIRXH"), population_size)
        self.set_compartment_mobility({
                    "X": False,
                    "H": False,
                })
        self.set_quadratic_processes([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])
        self.set_linear_processes([
                ("S", "H", containment_rate),
                ("I", "R", recovery_rate),
                ("I", "X", containment_rate+quarantine_rate),
            ])

class SEIRXModel(DeterministicEpiModel):
    """
    An SEIRX model derived from :class:`metapop.epi.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, symptomatic_rate, quarantine_rate, containment_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SEIRXH"), population_size)
        self.set_compartment_mobility({
                    "X": False,
                    "H": False,
                })
        self.set_quadratic_processes([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "E", +infection_rate),
            ])
        self.set_linear_processes([
                ("E", "I", symptomatic_rate),
                ("S", "H", containment_rate),
                ("I", "R", recovery_rate),
                ("I", "X", containment_rate+quarantine_rate),
            ])

class SIRSModel(DeterministicEpiModel):
    """
    An SIRS model derived from :class:`metapop.epi.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, waning_immunity_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SIR"), population_size)

        self.set_quadratic_processes([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])
        self.set_linear_processes([
                ("I", "R", recovery_rate),
                ("R", "S", waning_immunity_rate),
            ])

class SEIRModel(DeterministicEpiModel):
    """
    An SEIR model derived from :class:`metapop.epi.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, symptomatic_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SEIR"), population_size)

        self.set_quadratic_processes([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "E", +infection_rate),
            ])
        self.set_linear_processes([
                ("E", "I", symptomatic_rate),
                ("I", "R", recovery_rate),
            ])

class SEIRSModel(DeterministicEpiModel):
    """
    An SEIRS model derived from :class:`metapop.epi.DeterministicEpiModel`.
    """

    def __init__(self, R0, recovery_rate, symptomatic_rate, waning_immunity_rate, population_size=1.0):

        infection_rate = R0 * recovery_rate

        DeterministicEpiModel.__init__(self, list("SEIR"), population_size)

        self.set_quadratic_processes([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "E", +infection_rate),
            ])
        self.set_linear_processes([
                ("E", "I", symptomatic_rate),
                ("I", "R", recovery_rate),
                ("R", "S", waning_immunity_rate),
            ])


class SEAIHDRModel(DeterministicEpiModel):
    """
    An SEAIHDR model derived from :class:`metapop.epi.DeterministicEpiModel`.
    """

    def __init__(self, asymptomatic_infection_rate,
                       symptomatic_infection_rate, 
                       asymptomatic_rate, 
                       fraction_never_symptomatic,
                       rate_escaping_asymptomatic,
                       escape_rate,
                       fraction_requiring_ICU,
                       fraction_ICU_patients_succumbing,
                       rate_of_succumbing,
                       rate_leaving_ICU,
                       population_size=1.0):
        
        asymptomatic_recovery_rate = fraction_never_symptomatic*rate_escaping_asymptomatic
        symptomatic_rate = (1-fraction_never_symptomatic)*rate_escaping_asymptomatic

        hospitalization_rate = escape_rate*fraction_requiring_ICU
        symptomatic_recovery_rate = escape_rate*(1-fraction_requiring_ICU)

        fatality_rate = fraction_ICU_patients_succumbing*rate_of_succumbing
        discharge_rate = (1-fraction_ICU_patients_succumbing)*rate_leaving_ICU


        DeterministicEpiModel.__init__(self, list("SEAIHDR"), population_size)

        self.set_quadratic_processes([
                ("S", "A", "S", -asymptomatic_infection_rate),
                ("S", "A", "E", +asymptomatic_infection_rate),                
                ("S", "I", "S", -symptomatic_infection_rate),
                ("S", "I", "E", +symptomatic_infection_rate),
            ])
        self.set_linear_processes([
                ("E", "A", asymptomatic_rate),
                ("A", "I", symptomatic_rate),
                ("A", "R", asymptomatic_recovery_rate),
                ("I", "R", symptomatic_recovery_rate),
                ("I","H",hospitalization_rate),
                ("H","D",fatality_rate),
                ("H","R",discharge_rate),

            ])

if __name__=="__main__":
    epi = DeterministicEpiModel(list("SEIR"))
    print(epi.compartments)
    epi.set_compartment_mobility({
                "S": True,
                "E": True,
                "I": True,
                "R": True,
            })
    print("Mobility")
    print(epi.mobility)
    print()
    epi.set_linear_processes([
            ("E", "I", 1.0),
            ("I", "R", 1.0),
            ])
    print(epi.linear_rates)
    epi.set_quadratic_processes([
            ("S", "I", "S", -1.0),
            ("S", "I", "E", +1.0),
            ])
    print()
    for iM, M in enumerate(epi.quadratic_rates):
        print(epi.get_compartment(iM), M)

    import matplotlib.pyplot as pl

    N = 100
    epi = SISModel(R0=2,recovery_rate=1,population_size=N)
    epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
    tt = np.linspace(0,10,100)
    result = epi.get_result_dict(tt,['S','I'])

    pl.plot(tt, result['S'],label='S')
    pl.plot(tt, result['I'],label='I')
    pl.legend()

    pl.show()
