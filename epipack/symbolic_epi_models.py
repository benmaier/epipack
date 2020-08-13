"""
Provides an API to define NumericMatrixBased epidemiological models in terms of sympy symbolic expressions.
"""

import warnings

import numpy as np 
import scipy.sparse as sprs

import sympy

from epipack.integrators import integrate_dopri5, integrate_euler
from epipack.process_conversions import (
            processes_to_rates,
            transition_processes_to_rates,
            fission_processes_to_rates,
            fusion_processes_to_rates,
            transmission_processes_to_rates,
        )

from epipack.numeric_matrix_based_epi_models import NumericMatrixBasedEpiModel

from epipack.numeric_epi_models import NumericEpiModel

class SymbolicMixin():

    def ODEs(self):
        """
        Obtain the equations of motion for this model in form of equations.
        """
        t = sympy.symbols("t")
        Eqs = []
        ynew = self.dydt()
        for compartment, expr in zip(self.compartments,ynew):
            dXdt = sympy.Derivative(compartment, t)
            Eqs.append(sympy.Eq(dXdt, sympy.simplify(expr)))
        return Eqs

    def ODEs_jupyter(self):
        """
        Pretty-print the equations of motion for this model in a Jupyter notebook.
        """

        from IPython.display import Math, display
        for ode in self.ODEs():
            display(Math(sympy.latex(ode)))

    def find_fixed_points(self):
        """
        Find states in which this model is stationary (fixed points).
        """
        ynew = [expr for expr in self.dydt()]
        return sympy.nonlinsolve(ynew, self.compartments)


    def jacobian(self,simplify=True):
        """
        Obtain the Jacobian for this model.
        """

        try:
            self.linear_rates
            has_matrix = True
        except AttributeError as e:
            has_matrix = False

        if has_matrix and not self.has_functional_rates:
            y = sympy.Matrix(self.compartments)
            J = sympy.Matrix(self.linear_rates) 
            
            for i in range(self.N_comp):
                J[i,:] += (self.quadratic_rates[i] * y + self.quadratic_rates[i].T * y).T

        else:
            y = sympy.Matrix(self.compartments)
            J = sympy.zeros(self.N_comp, self.N_comp)
            dydt = self.dydt()
            for i in range(self.N_comp):
                for j in range(self.N_comp):
                    J[i,j] = sympy.diff(dydt[i], self.compartments[j])

        if simplify:
            J = sympy.simplify(J)

        return J

    def get_jacobian_at_fixed_point(self,fixed_point_dict,simplify=True):
        """
        Obtain the Jacobian at a given fixed point.

        Parameters
        ----------
        fixed_point_dict : dict
            A dictionary where a compartment symbol maps to an expression
            (the value of this compartment in the fixed point). 
            If compartments are missing, it is implicitly assumed
            that this compartment has a value of zero.
        simplify : bool
            whether or not to let sympy try to simplify the expressions

        Returns
        -------
        J : sympy.Matrix
            The Jacobian matrix at the given fixed point.

        """
        
        fixed_point = self._convert_fixed_point_dict(fixed_point_dict)

        J = self.jacobian(False)
        
        for compartment, value in fixed_point:
            J = J.subs(compartment, value)

        if simplify:
            J = sympy.simplify(J)

        return J

    def get_eigenvalues_at_fixed_point(self,fixed_point_dict):
        """
        Obtain the Jacobian's eigenvalues at a given fixed point.

        Parameters
        ----------
        fixed_point_dict : dict
            A dictionary where a compartment symbol maps to an expression
            (the value of this compartment in the fixed point). 
            If compartments are missing, it is implicitly assumed
            that this compartment has a value of zero.

        Returns
        -------
        eigenvalues : dict
            Each entry maps an eigenvalue expression to its multiplicity.

        """
        J = self.get_jacobian_at_fixed_point(fixed_point_dict)
        return J.eigenvals()
        

    def get_eigenvalues_at_disease_free_state(self,disease_free_state=None):
        """
        Obtain the Jacobian's eigenvalues at the disease free state.

        Parameters
        ----------
        disease_free_state : dict, default = None
            A dictionary where a compartment symbol maps to an expression
            (the value of this compartment in the fixed point). 
            If compartments are missing, it is implicitly assumed
            that this compartment has a value of zero.

            If ``None``, the disease_free_state is assumed to be at
            ``disease_free_state = { S: 1 }``.

        Returns
        -------
        eigenvalues : dict
            Each entry maps an eigenvalue expression to its multiplicity.

        """

        if disease_free_state is None:
            S = sympy.symbols("S")
            if S not in self.compartments:
                raise ValueError("Disease free state was not provided to the method. I tried to assume the disease free state is at S = 1, but no `S`-compartment was found.")
            disease_free_state = {S:1}

        return self.get_eigenvalues_at_fixed_point(disease_free_state)

    def _convert_fixed_point_dict(self,fixed_point_dict):
        """
        Get a fixed point list.

        Parameters
        ----------
        fixed_point_dict : dict
            A dictionary where a compartment symbol maps to an expression
            (the value of this compartment in the fixed point). 
            If compartments are missing, it is implicitly assumed
            that this compartment has a value of zero.

        Returns
        -------
        fixed_point : iterator of list of tuple
            A list that is ``N_comp`` entries long.
            Each entry contains a compartment symbol and 
            an expression that corresponds to the value
            this compartment assumes in the fixed point.
        """
        fixed_point = {}
        for compartment in self.compartments:
            fixed_point[compartment] = 0
        fixed_point.update(fixed_point_dict)
        return fixed_point.items()

    def set_parameter_values(self,parameter_values):
        """
        Set numerical values for the parameters of this model

        Parameters
        ==========
        parameter_values : dict
            A dictionary mapping compartment symbols to
            numerical values.
        """
        self.parameter_values = parameter_values
        
    def get_numerical_dydt(self):
        r"""
        Returns values of the given compartments at the demanded
        time points (as a numpy.ndarray of shape 
        ``(return_compartments), len(time_points)``.

        Parameters
        ==========
        time_points : np.ndarray
            An array of time points at which the compartment values
            should be evaluated and returned.
        return_compartments : list, default = None
            A list of compartments for which the result should be returned.
            If ``return_compartments`` is None, all compartments will
            be returned.
        integrator : str, default = 'dopri5'
            Which method to use for integration. Currently supported are
            ``'euler'`` and ``'dopri5'``. If ``'euler'`` is chosen,
            :math:`\delta t` will be determined by the difference
            of consecutive entries in ``time_points``.
        adopt_final_state : bool, default = False
            Whether or not to adopt the final state of the integration
        """

        these_symbols = [sympy.symbols("t")] + self.compartments

        params = list(self.parameter_values.items())
        param_symbols = set([_p[0] for _p in params])

        odes = [ ode.subs(params) for ode in self.dydt() ]

        not_set = []
        for ode in odes:
            not_set.extend(ode.free_symbols)

        not_set = set(not_set) - set(these_symbols)

        if len(not_set) > 0:
            raise ValueError("Parameters", set(not_set), "have not been set. Please set them using",
                             "SymbolicEpiModel.parameter_values()")

        F_sympy = sympy.lambdify(these_symbols, odes)

        def dydt(t, y, *args, **kwargs):
            these_args = [t] + y.tolist()
            return np.array(F_sympy(*these_args))

        return dydt

    def get_numerical_event_and_rate_functions(self):
        rates = self.birth_rate_functions + self.linear_rate_functions
        if self.correct_for_dynamical_population_size:
            population_size = sum(self.compartments)
        else:
            population_size = self.initial_population_size
        rates += [ r/population_size for r in self.quadratic_rate_functions ]
        events = self.birth_event_updates + self.linear_event_updates + self.quadratic_event_updates
        
        these_symbols = [sympy.symbols("t")] + self.compartments

        params = list(self.parameter_values.items())
        param_symbols = set([_p[0] for _p in params])

        rates = [ sympy.sympify(rate).subs(params) for rate in rates ]
        events = [ np.array(event).astype(np.float64).flatten() for event in events ]

        not_set = []
        for rate in rates:
            not_set.extend(rate.free_symbols)

def get_temporal_interpolation(time_data, value_data, interpolation_degree=1):
    """
    Obtain a symbolic piecewise function that interpolates between values
    given in ``value_data`` for the intervals defined in ``time_data``,
    based on a spline interpolation of degree ``interpolation_degree``.
    If ``interpolation_degree == 0``, the function changes according to step
    functions. In this case ``time_data`` needs to have one value more than
    ``value_data``.
    """
    t = sympy.symbols("t")
    if interpolation_degree == 0:
        if len(time_data) != len(value_data)+1:
            raise ValueError("For ``interpolation_degree == 0``, `time_data`` needs to have one value more than ``value_data``.")
        return sympy.Piecewise(*[
                    (v, (time_data[i] <= t) & ( t < time_data[i+1])) \
                    for i, v in enumerate(value_data)
                ])
    else:
        return sympy.interpolating_spline(interpolation_degree, t, time_data, value_data)


class SymbolicMatrixBasedEpiModel(SymbolicMixin, NumericMatrixBasedEpiModel):
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
        
        >>> epi = SymbolicEpiModel(symbols("S I R"))
        >>> print(epi.compartments)
        [ S, I, R ]


    """

    def __init__(self,compartments,initial_population_size=1,correct_for_dynamical_population_size=False):
        """
        """
        NumericMatrixBasedEpiModel.__init__(self, compartments, initial_population_size, correct_for_dynamical_population_size)


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
            

class SymbolicEpiModel(SymbolicMixin, NumericEpiModel):

    def __init__(self,compartments,
                      initial_population_size=1,
                      correct_for_dynamical_population_size=False,
                      ):
        NumericEpiModel.__init__(self,
                      compartments,
                      initial_population_size=initial_population_size,
                      correct_for_dynamical_population_size=correct_for_dynamical_population_size,
                )
        self.parameter_values = {}

    def _check_rate_for_functional_dependency(self,rate):
        try:
            t = sympy.symbols("t")
            has_time_dependence = t in rate.free_symbols
            self.rates_have_explicit_time_dependence |= has_time_dependence
        except AttributeError as e:
            return

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
            linear_event_updates = self.linear_event_updates
            birth_event_updates = self.birth_event_updates
            linear_rate_functions = self.linear_rate_functions
            birth_rate_functions = self.birth_rate_functions

        for acting_compartments, rate, affected_compartments in event_list:

            dy = sympy.zeros(self.N_comp,1)
            for trg, change in affected_compartments:
                _t = self.get_compartment_id(trg)
                dy[_t] = change

            if acting_compartments[0] is None:
                self._check_rate_for_functional_dependency(rate)
                birth_event_updates.append( dy )
                birth_rate_functions.append( rate )
            else:
                _s = self.get_compartment_id(acting_compartments[0])                
                self._check_rate_for_functional_dependency(rate)
                this_rate = rate * acting_compartments[0]
                linear_event_updates.append( dy )
                linear_rate_functions.append( this_rate )


        if not allow_nonzero_column_sums and len(linear_rate_functions)>0:
            _y = np.ones(self.N_comp)
            test = sympy.zeros(self.N_comp,1)
            for dy, r in zip (linear_event_updates, linear_rate_functions):
                test += dy * r
            for dy, r in zip (birth_event_updates, birth_rate_functions):
                test += dy * r
            test_sum = sum(test)
            if test_sum != 0:
                warnings.warn("events do not sum to zero for each column:" + str(test_sum))

        self.linear_event_updates = linear_event_updates
        self.linear_rate_functions = linear_rate_functions
        self.birth_event_updates = birth_event_updates
        self.birth_rate_functions = birth_rate_functions

        return self

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

            _s0 = coupling_compartments[0]
            _s1 = coupling_compartments[1]

            dy = sympy.zeros(self.N_comp,1)
            for trg, change in affected_compartments:
                _t = self.get_compartment_id(trg)
                dy[_t] = change

            self._check_rate_for_functional_dependency(rate)
            this_rate = rate * _s0 * _s1

            quadratic_event_updates.append( dy )
            quadratic_rate_functions.append( this_rate )

        if not allow_nonzero_column_sums and len(quadratic_rate_functions)>0:
            dy = sympy.zeros(self.N_comp,1)
            test = sympy.zeros(self.N_comp,1)
            for dy, r in zip (quadratic_event_updates, quadratic_rate_functions):
                test += dy * r 
            test_sum = sum(test)
            if np.abs(test_sum) > 1e-15:
                warnings.warn("events do not sum to zero for each column:" + str(test_sum))

        self.quadratic_event_updates = quadratic_event_updates
        self.quadratic_rate_functions = quadratic_rate_functions

        return self

    def dydt(self):
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
        ynew = sympy.zeros(self.N_comp,1)

        for dy, r in zip(self.birth_event_updates, self.birth_rate_functions):
            ynew += r * dy

        for dy, r in zip(self.linear_event_updates, self.linear_rate_functions):
            ynew += r * dy

        if self.correct_for_dynamical_population_size:
            population_size = sum(self.compartments)
        else:
            population_size = self.initial_population_size

        for dy, r in zip(self.quadratic_event_updates, self.quadratic_rate_functions):
            ynew += r/population_size * dy 

        return ynew

class SymbolicSIModel(SymbolicEpiModel):
    """
    An SI model derived from :class:`epipack.symbolic_epi_models.SymbolicEpiModel`.
    """

    def __init__(self, infection_rate, initial_population_size=1):

        S, I = sympy.symbols("S I")

        SymbolicEpiModel.__init__(self,[S, I], initial_population_size)

        self.set_processes([
                (S, I, infection_rate, I, I),
            ])

class SymbolicSIRModel(SymbolicEpiModel):
    """
    An SIR model derived from :class:`epipack.symbolic_epi_models.SymbolicEpiModel`.
    """

    def __init__(self, infection_rate, recovery_rate, initial_population_size=1):

        S, I, R = sympy.symbols("S I R")

        SymbolicEpiModel.__init__(self,[S, I, R], initial_population_size)

        self.add_transmission_processes([
                (S, I, infection_rate, I, I),
            ])

        self.add_transition_processes([
                (I, recovery_rate, R),
            ])

class SymbolicSISModel(SymbolicEpiModel):
    """
    An SIS model derived from :class:`epipack.symbolic_epi_models.SymbolicEpiModel`.
    """

    def __init__(self, infection_rate, recovery_rate, initial_population_size=1):

        S, I = sympy.symbols("S I")

        SymbolicEpiModel.__init__(self,[S, I], initial_population_size)

        self.add_transmission_processes([
                (S, I, infection_rate, I, I),
            ])

        self.add_transition_processes([
                (I, recovery_rate, S),
            ])

class SymbolicSIRSModel(SymbolicEpiModel):
    """
    An SIRS model derived from :class:`epipack.symbolic_epi_models.SymbolicEpiModel`.
    """

    def __init__(self, infection_rate, recovery_rate, waning_immunity_rate, initial_population_size=1):

        S, I, R = sympy.symbols("S I R")

        SymbolicEpiModel.__init__(self,[S, I, R], initial_population_size)

        self.add_transmission_processes([
                (S, I, infection_rate, I, I),
            ])

        self.add_transition_processes([
                (I, recovery_rate, R),
                (R, waning_immunity_rate, S),
            ])

if __name__=="__main__":    # pragma: no cover
    eta, rho = sympy.symbols("eta rho")
    epi = SymbolicSIRModel(eta, rho)

    #for C, M in zip(epi.compartments, epi.quadratic_rates):
    #    print(C, M)

    print(epi.dydt())
    
    print(epi.jacobian())

    print(epi.get_eigenvalues_at_disease_free_state())

    print(epi.dydt())

    print(epi.ODEs())

    print(epi.find_fixed_points())

    epi = SymbolicSISModel(eta, rho)
    print()
    print(epi.ODEs())
    print(epi.find_fixed_points())

    omega = sympy.symbols("omega")
    epi = SymbolicSIRSModel(eta, rho, omega)
    print()
    print(epi.ODEs())
    print(epi.find_fixed_points())


    import sympy
    from epipack import SymbolicEpiModel

    S, I, eta, rho = sympy.symbols("S I eta rho")

    SIS = SymbolicEpiModel([S,I])
    SIS.add_transmission_processes([
            (I, S, eta, I, I),
        ])
    SIS.add_transition_processes([
            (I, rho, S),
        ])

    print(SIS.find_fixed_points())

    print(SIS.get_eigenvalues_at_fixed_point({S:1}))

    print("==========")
    SIS = SymbolicEpiModel([S,I])
    SIS.set_processes([
            (I, S, eta/(1-I), I, I),
            (I, rho, S),
        ])
    print(SIS.jacobian())
    print(SIS.get_eigenvalues_at_disease_free_state())


    print("gray scott")
    u, v, f, k = sympy.symbols("u v f k")
    GS = SymbolicMatrixBasedEpiModel([u,v])
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


    N = sympy.symbols("N")
    epi = SymbolicSIRSModel(eta, rho, omega, initial_population_size=N)
    print()
    print(epi.ODEs())
    print(epi.find_fixed_points())

    print("==========")
    x = sympy.symbols("x")
    SIS = SymbolicEpiModel([x,I])
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
    epi = SymbolicSISModel(eta, rho)

    epi.set_initial_conditions({S: 1-0.01, I:0.01 })
    epi.set_parameter_values({eta:2,rho:1})

    tt = np.linspace(0,10,1000)
    result = epi.integrate(tt)

    import matplotlib.pyplot as pl

    pl.plot(tt, result[S])
    pl.plot(tt, result[I])

    print("===========")
    t, S, I = sympy.symbols("t S I")
    epi = SymbolicSISModel((1.5+sympy.cos(t))*eta, rho)

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
    epi = SymbolicSISModel(eta/(1-S), rho)

    epi.set_initial_conditions({S: 1-0.2, I:0.2 })
    epi.set_parameter_values({eta:2,rho:1})

    tt = np.linspace(0,5,1000)
    result = epi.integrate(tt)

    import matplotlib.pyplot as pl

    pl.figure()
    pl.plot(tt, result[S])
    pl.plot(tt, result[I])
    pl.show()

