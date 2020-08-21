"""
Provides an API to define epidemiological models in terms of sympy symbolic expressions based on a matrix description.
"""

import warnings

import numpy as np 
import scipy.sparse as sprs

import sympy

from epipack.process_conversions import (
            processes_to_rates,
            transition_processes_to_rates,
            fission_processes_to_rates,
            fusion_processes_to_rates,
            transmission_processes_to_rates,
        )

from epipack.numeric_matrix_epi_models import MatrixEpiModel
from epipack.symbolic_epi_models import SymbolicMixin

class SymbolicMatrixEpiModel(SymbolicMixin,MatrixEpiModel):
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
    linear_rates : sympy.Matrix
        Matrix containing
        transition rates of the linear processes.
    quadratic_rates : list of sympy.Matrix
        List of matrices that contain
        transition rates of the quadratic processes
        for each compartment.
    affected_by_quadratic_process : :obj:`list` of :obj:`int`
        List of integer compartment IDs, collecting
        compartments that are affected
        by the quadratic processes

    Example
    -------

    .. code:: python

        >>> epi = SymbolicMatrixEpiModel(symbols("S I R"))
        >>> print(epi.compartments)
        [ S, I, R ]

    """

    def __init__(self,compartments,initial_population_size=1,correct_for_dynamical_population_size=False):
        """
        """
        MatrixEpiModel.__init__(self, compartments, initial_population_size, correct_for_dynamical_population_size)


        self.t = sympy.symbols("t")
        if self.t in self.compartments:
            raise ValueError("Don't use `t` as a compartment symbol, as it is reserved for time.")

        self.has_functional_rates = False

        self.birth_rates = sympy.zeros(self.N_comp,1)
        self.linear_rates = sympy.zeros(self.N_comp, self.N_comp)
        self.quadratic_rates = [ sympy.zeros(self.N_comp, self.N_comp)\
                                 for c in range(self.N_comp) ]
        self.birth_events = sympy.zeros(self.N_comp,1)
        self.linear_events = sympy.zeros(self.N_comp, self.N_comp)
        self.quadratic_events = [ sympy.zeros(self.N_comp, self.N_comp)\
                                 for c in range(self.N_comp) ]
        self.parameter_values = {}

    def set_linear_rates(self,rate_list,reset_rates=True,allow_nonzero_column_sums=True):
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
            This keyword has no function in this class
        reset_rates : bool, default : True
            Whether to reset all linear rates to zero before
            setting the new ones.
        """

        if reset_rates:
            linear_rates = sympy.zeros(self.N_comp, self.N_comp)
            birth_rates = sympy.zeros(self.N_comp,1)
            self.has_functional_rates = False
        else:
            linear_rates = sympy.Matrix(self.linear_rates)
            birth_rates = sympy.Matrix(self.birth_rates)

        for acting_compartment, affected_compartment, rate in rate_list:

            _t = self.get_compartment_id(affected_compartment)
            if acting_compartment is None:
                birth_rates[_t] += rate
            else:
                _s = self.get_compartment_id(acting_compartment)
                linear_rates[_t, _s] += rate

            self._check_rate_for_functional_dependency(rate)

        self.linear_rates = linear_rates
        self.birth_rates = birth_rates

        return self

    def _check_rate_for_functional_dependency(self,rate):
        try:
            self.has_functional_rates |= any([ compartment in rate.free_symbols for compartment in self.compartments])
        except AttributeError as e:
            return


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
            This keyword has no function in this class
        reset_rates : bool, default : True
            Whether to reset all quadratic rates to zero before
            setting the new ones.

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
                matrices[c] = sympy.zeros(self.N_comp, self.N_comp)

            all_affected = []
            self.has_functional_rates = False
        else:
            matrices =  [ sympy.Matrix(M) for M in self.quadratic_rates ]
            all_affected = self.affected_by_quadratic_process if len(self.affected_by_quadratic_process)>0 else []

        for coupling0, coupling1, affected, rate in rate_list:

            c0, c1 = sorted([ self.get_compartment_id(c) for c in [coupling0, coupling1] ])
            a = self.get_compartment_id(affected)

            self._check_rate_for_functional_dependency(rate)

            matrices[a][c0,c1] += rate
            all_affected.append(a)

        self.affected_by_quadratic_process = sorted(list(set(all_affected)))
        self.quadratic_rates = matrices

        return self


    def dydt(self):
        """
        Obtain the equations of motion for this model in form of a sympy.Matrix.
        """

        y = sympy.Matrix(self.compartments)

        ynew = self.linear_rates * y + self.birth_rates
        if self.correct_for_dynamical_population_size:
            population_size = sum(self.compartments)
        else:
            population_size = self.initial_population_size
        for c in self.affected_by_quadratic_process:
            ynew[c] += (y.T * self.quadratic_rates[c] * y)[0,0] / population_size

        return ynew

class SymbolicMatrixSIModel(SymbolicMatrixEpiModel):
    """
    An SI model derived from :class:`epipack.symbolic_epi_models.SymbolicMatrixEpiModel`.
    """

    def __init__(self, infection_rate, initial_population_size=1):

        S, I = sympy.symbols("S I")

        SymbolicMatrixEpiModel.__init__(self,[S, I], initial_population_size)

        self.set_processes([
                (S, I, infection_rate, I, I),
            ])

class SymbolicMatrixSIRModel(SymbolicMatrixEpiModel):
    """
    An SIR model derived from :class:`epipack.symbolic_epi_models.SymbolicMatrixEpiModel`.
    """

    def __init__(self, infection_rate, recovery_rate, initial_population_size=1):

        S, I, R = sympy.symbols("S I R")

        SymbolicMatrixEpiModel.__init__(self,[S, I, R], initial_population_size)

        self.add_transmission_processes([
                (S, I, infection_rate, I, I),
            ])

        self.add_transition_processes([
                (I, recovery_rate, R),
            ])

class SymbolicMatrixSISModel(SymbolicMatrixEpiModel):
    """
    An SIS model derived from :class:`epipack.symbolic_epi_models.SymbolicMatrixEpiModel`.
    """

    def __init__(self, infection_rate, recovery_rate, initial_population_size=1):

        S, I = sympy.symbols("S I")

        SymbolicMatrixEpiModel.__init__(self,[S, I], initial_population_size)

        self.add_transmission_processes([
                (S, I, infection_rate, I, I),
            ])

        self.add_transition_processes([
                (I, recovery_rate, S),
            ])

class SymbolicMatrixSIRSModel(SymbolicMatrixEpiModel):
    """
    An SIRS model derived from :class:`epipack.symbolic_epi_models.SymbolicMatrixEpiModel`.
    """

    def __init__(self, infection_rate, recovery_rate, waning_immunity_rate, initial_population_size=1):

        S, I, R = sympy.symbols("S I R")

        SymbolicMatrixEpiModel.__init__(self,[S, I, R], initial_population_size)

        self.add_transmission_processes([
                (S, I, infection_rate, I, I),
            ])

        self.add_transition_processes([
                (I, recovery_rate, R),
                (R, waning_immunity_rate, S),
            ])

if __name__=="__main__":    # pragma: no cover

    import sympy

    S, I, eta, rho = sympy.symbols("S I eta rho")

    SIS = SymbolicMatrixEpiModel([S,I])
    SIS.add_transmission_processes([
            (I, S, eta, I, I),
        ])
    SIS.add_transition_processes([
            (I, rho, S),
        ])

    print(SIS.find_fixed_points())

    print(SIS.get_eigenvalues_at_fixed_point({S:1}))

    print("==========")
    SIS = SymbolicMatrixEpiModel([S,I])
    SIS.set_processes([
            (I, S, eta/(1-I), I, I),
            (I, rho, S),
        ])
    print(SIS.jacobian())
    print(SIS.get_eigenvalues_at_disease_free_state())


    print("gray scott")
    u, v, f, k = sympy.symbols("u v f k")
    GS = SymbolicMatrixEpiModel([u,v])
    GS.set_linear_rates([
            (None, u, f),
            (u, u, -f),
            (v, v, -f-k),
        ])

    GS.set_quadratic_rates([
            (u, v, u, -v),
            (u, v, v, +v),
        ])


    GS.set_processes([
            (u, f, None),
            (None, f, u),
            (v, f+k, None),
            (u, v, v*1, v, v),
        ],ignore_rate_position_checks=True)

    print(GS.ODEs())

    print(GS.find_fixed_points())

    print("===========")


    N, omega = sympy.symbols("N omega")

    epi = SymbolicMatrixSIRSModel(eta, rho, omega, initial_population_size=N)
    print()
    print(epi.ODEs())
    print(epi.find_fixed_points())

    print("==========")
    x = sympy.symbols("x")
    SIS = SymbolicMatrixEpiModel([x,I])
    SIS.set_processes([
            (I, x, eta/(1-I), I, I),
            (I, rho, x),
        ])
    try:
        print(SIS.get_eigenvalues_at_disease_free_state())
    except ValueError as e:
        print(e)

    print("===========")
    S, I = sympy.symbols("S I")
    epi = SymbolicMatrixSISModel(eta, rho)

    epi.set_initial_conditions({S: 1-0.01, I:0.01 })
    epi.set_parameter_values({eta:2,rho:1})

    tt = np.linspace(0,10,1000)
    result = epi.integrate(tt)

    import matplotlib.pyplot as pl

    pl.plot(tt, result[S])
    pl.plot(tt, result[I])

    print("===========")
    t, S, I = sympy.symbols("t S I")
    epi = SymbolicMatrixSISModel((1.5+sympy.cos(t))*eta, rho)

    epi.set_initial_conditions({S: 1-0.2, I:0.2 })
    epi.set_parameter_values({eta:2,rho:1})

    tt = np.linspace(0,20,1000)
    result = epi.integrate(tt)

    import matplotlib.pyplot as pl

    pl.figure()
    pl.plot(tt, result[S])
    pl.plot(tt, result[I])

    print("===========")
    t, S, I = sympy.symbols("t S I")
    epi = SymbolicMatrixSISModel(eta/(1-S), rho)

    epi.set_initial_conditions({S: 1-0.2, I:0.2 })
    epi.set_parameter_values({eta:2,rho:1})

    tt = np.linspace(0,5,1000)
    result = epi.integrate(tt)

    import matplotlib.pyplot as pl

    pl.figure()
    pl.plot(tt, result[S])
    pl.plot(tt, result[I])
    pl.show()

