"""
Provides an API to define epidemiological
models in terms of sympy symbolic expressions.
"""

import warnings

import numpy as np
import scipy.sparse as sprs

import sympy

from epipack.numeric_epi_models import EpiModel, custom_choice
from IPython.display import Math, display
from sympy.printing.theanocode import theano_function

class SymbolicMixin():
    """
    Provides methods that are useful to both
    :class:`epipack.symbolic_epi_models.SymbolicEpiModel`
    and
    :class:`epipack.symbolic_matrix_epi_models.SymbolicMatrixEpiModel`
    """

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

        Parameters
        ----------
        simplify : bool, default = True
            If ``True``, `epipack` will try to simplify
            the evaluated Jacobian. This might not be
            desirable in some cases due to its
            long evaluation time, which is why
            it can be turned off.
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
                raise ValueError("The disease free state was not provided to the method. "+\
                                 "I tried to assume the disease free state is at S = 1, "+\
                                 "but no `S`-compartment was found.")
            disease_free_state = {S:1}

        return self.get_eigenvalues_at_fixed_point(disease_free_state)

    def _convert_fixed_point_dict(self,fixed_point_dict):
        """
        Get a fixed point item iterator.

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
        Set numerical values for the parameters of this model.
        This might include free symbols that are part of symbolized
        rate functions.

        Parameters
        ==========
        parameter_values : dict
            A dictionary mapping symbols to
            numerical values.
        """
        self.parameter_values = parameter_values

    def get_numerical_dydt(self,lambdify_modules='numpy'):
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

        Returns
        =======
        dydt : func
            A function ``dydt(t, y, *args, **kwargs)`` that returns
            the numerical momenta of this system at time ``t`` and
            state vector ``y``.
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

        F_sympy = sympy.lambdify(these_symbols, odes, modules=lambdify_modules)

        def dydt(t, y, *args, **kwargs):
            these_args = [t] + y.tolist()
            return np.array(F_sympy(*these_args))

        return dydt

    def get_numerical_event_and_rate_functions(self):
        """
        Converts the symbolic event lists and corresponding
        symbolic rates to functions that return numeric
        event lists and numeric rates based on the current
        time and state vector.

        This function is needed in the
        :class:`epipack.numeric_epi_models.EpiModel`
        base class for stochastic simulations.

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

        not_set = set(not_set) - set(these_symbols)

        if len(not_set) > 0:
            raise ValueError("Parameters " + str(set(not_set)) +\
                             "have not been set. Please set them using " +\
                             "SymbolicEpiModel.parameter_values()")

        sympy_rates = sympy.lambdify(these_symbols, rates)

        def get_event_rates(t, y):
            """
            Returns numerical event rate lists based on current
            state vector ``y`` and time ``t``.
            """
            these_args = [t] + y.tolist()
            return np.array(sympy_rates(*these_args))

        def get_compartment_changes(rates):
            """
            Choose an event change vector based on the current ``rates``.
            """
            idy = custom_choice(rates/rates.sum())
            return events[idy]

        return get_event_rates, get_compartment_changes

def get_temporal_interpolation(time_data, value_data, interpolation_degree=1):
    """
    Obtain a symbolic piecewise function that interpolates between values
    given in ``value_data`` for the intervals defined in ``time_data``,
    based on a spline interpolation of degree ``interpolation_degree``.
    If ``interpolation_degree == 0``, the function changes according to step
    functions. In this case ``time_data`` needs to have one value more than
    ``value_data``.

    The values in ``time_data`` and ``value_data`` can be symbols or numeric
    values.

    Parameters
    ----------
    time_data : list
        Sorted list of time values.
    value_data : list
        List of values corresponding to the times given in ``time_data``.
    interpolation_degree : int
        The degree of the polynomial that interpolates between values.
    """
    t = sympy.symbols("t")
    if interpolation_degree == 0:
        if len(time_data) != len(value_data)+1:
            raise ValueError("For ``interpolation_degree == 0``, `time_data`` " +\
                             "needs to have one value more than ``value_data``.")
        return sympy.Piecewise(*[
                    (v, (time_data[i] <= t) & ( t < time_data[i+1])) \
                    for i, v in enumerate(value_data)
                ])
    else:
        return sympy.interpolating_spline(interpolation_degree, t, time_data, value_data)

class SymbolicEpiModel(SymbolicMixin, EpiModel):
    """
    Define a model based on the analytical framework
    offered by `Sympy <https://sympy.org/>`_.

    This class uses the event-based framework
    where state-change vectors are associated with
    event rates.

    Parameters
    ==========
    compartments : list
        A list of :class:`sympy.Symbol` instances that
        symbolize compartments.
    initial_population_size : float, default = 1.0
        The population size at :math:`t = 0`.
    correct_for_dynamical_population_size : bool, default = False
        If ``True``, the quadratic coupling terms will be
        divided by the sum of all compartments, otherwise they
        will be divided by the initial population size.

    Attributes
    ==========
    compartments : list
        A list of :class:`sympy.Symbol` instances that
        symbolize compartments.
    N_comp : :obj:`int`
        Number of compartments (including population number)
    parameter_values : dict
        Maps parameter symbols to numerical values,
    initial_population_size : float
        The population size at :math:`t = 0`.
    correct_for_dynamical_population_size : bool
        If ``True``, the quadratic coupling terms will be
        divided by the sum of all compartments, otherwise they
        will be divided by the initial population size.
    birth_rate_functions : list of symbolic expressions
        A list of functions that return rate values based on time ``t``
        and state vector ``y``. Each entry corresponds to an event update
        in ``self.birth_event_updates``.
    birth_event_updates : list of sympy.Matrix
        A list of vectors. Each entry corresponds to a rate in
        ``birth_rate_functions`` and quantifies the change in
        individual counts in the compartments.
    linear_rate_functions : list of symbolic expressions
        A list of functions that return rate values based on time ``t``
        and state vector ``y``. Each entry corresponds to an event update
        in ``self.linear_event_updates``.
    linear_event_updates : list of sympy.Matrix
        A list of vectors. Each entry corresponds to a rate in
        ``linear_rate_functions`` and quantifies the change in
        individual counts in the compartments.
    quadratic_rate_functions : list of symbolic expressions
        A list of functions that return rate values based on time ``t``
        and state vector ``y``. Each entry corresponds to an event update
        in ``self.quadratic_event_updates``.
    quadratic_event_updates : list of sympy.Matrix
        A list of vectors. Each entry corresponds to a rate in
        ``quadratic_rate_functions`` and quantifies the change in
        individual counts in the compartments.
    y0 : numpy.ndarray
        The initial conditions.
    rates_have_explicit_time_dependence : bool
        Internal switch that's flipped when a non-constant
        rate is passed to the model.


    """

    def __init__(self,compartments,
                      initial_population_size=1,
                      correct_for_dynamical_population_size=False,
                      ):

        EpiModel.__init__(self,
                      compartments,
                      initial_population_size=initial_population_size,
                      correct_for_dynamical_population_size=correct_for_dynamical_population_size,
                )

        self.t = sympy.symbols("t")
        if self.t in self.compartments:
            raise ValueError("Don't use `t` as a compartment symbol, as it is reserved for time.")

        self.parameter_values = {}

    def _check_rate_for_functional_dependency(self,rate):
        """
        Sets the attribute ``rates_have_explicit_time_dependence``
        to ``True`` if ``rate`` has an explicit functional dependency
        on a :class:`sympy.Symbol` called ``"t"``.
        """
        try:
            t = sympy.symbols("t")
            has_time_dependence = t in rate.free_symbols
            self.rates_have_explicit_time_dependence |= has_time_dependence
        except AttributeError as e:
            return

    def set_linear_events(self,event_list,allow_nonzero_column_sums=False,reset_events=True):
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
        else:
            linear_event_updates = self.linear_event_updates
            birth_event_updates = self.birth_event_updates
            linear_rate_functions = self.linear_rate_functions
            birth_rate_functions = self.birth_rate_functions

        for acting_compartments, rate, affected_compartments in event_list:

            dy = sympy.zeros(self.N_comp,1)
            for trg, change in affected_compartments:
                _t = self.get_compartment_id(trg)
                dy[_t] += change

            if acting_compartments[0] is None:
                self._check_rate_for_functional_dependency(rate)
                birth_event_updates.append( dy )
                birth_rate_functions.append( rate )
            else:
                # check if compartment was defined as a function
                this_compartment = acting_compartments[0]

                self._check_rate_for_functional_dependency(rate)
                this_rate = rate * this_compartment
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

    def set_quadratic_events(self,event_list,allow_nonzero_column_sums=False,reset_events=True):
        r"""
        Define the quadratic transition events between compartments.

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
        else:
            quadratic_event_updates = self.quadratic_event_updates
            quadratic_rate_functions = self.quadratic_rate_functions

        for coupling_compartments, rate, affected_compartments in event_list:

            _s0 = coupling_compartments[0]
            _s1 = coupling_compartments[1]

            dy = sympy.zeros(self.N_comp,1)
            for trg, change in affected_compartments:
                _t = self.get_compartment_id(trg)
                dy[_t] += change

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
            if test_sum != 0:
                warnings.warn("events do not sum to zero for each column:" + str(test_sum))

        self.quadratic_event_updates = quadratic_event_updates
        self.quadratic_rate_functions = quadratic_rate_functions

        return self

    def dydt(self):
        """
        Compute the momenta of the epidemiological model as
        symbolic expressions.

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

class SymbolicODEModel(SymbolicEpiModel):
    """
    Define a model purely based on a list of ODEs.

    Parameters
    ==========
    ODEs : list
        A list of symbolic ODEs in format

        .. code:: python

            sympy.Eq(sympy.Derivative(Y, t), expr)
    """

    def __init__(self, ODEs):

        compartments = self._get_compartments_and_assert_derivatives(ODEs)
        self._dydt = [ eq.rhs for eq in ODEs ]

        SymbolicEpiModel.__init__(self, compartments)

    def _get_compartments_and_assert_derivatives(self, ODEs):
        """
        Iterate through a list of ODES, assert that
        each of them is of format 

        .. code:: python

            sympy.Eq(sympy.Derivative(Y, t), expr)

        and return a list of `Y` symbols that
        are assumed to be compartments (ordered
        by appearance in ``ODEs``).
        """

        t = sympy.symbols("t")
        compartments = []

        for eq in ODEs:

            lhs, rhs = eq.lhs, eq.rhs    
            assert(isinstance(lhs, sympy.Derivative))

            expr = lhs.expr
            variables = lhs.variables
            assert(len(expr.free_symbols) == 1)
            assert(len(variables) == 1)
            assert(variables[0] == t)
            compartments.append(list(expr.free_symbols)[0])

        return compartments

    def dydt(self):
        """
        Return the momenta of the epidemiological model as
        symbolic expressions.
        """

        return self._dydt

    def simulate(self,*args,**kwargs):
        raise AttributeError("'SymbolicODEModel' object has no attribute 'simulate'")

    def set_linear_events(self,*args,**kwargs):
        raise AttributeError("'SymbolicODEModel' object has no attribute 'set_linear_events'")

    def set_quadratic_events(self,*args,**kwargs):
        raise AttributeError("'SymbolicODEModel' object has no attribute 'set_quadratic_events'")

    def set_processes(self,*args,**kwargs):
        raise AttributeError("'SymbolicODEModel' object has no attribute 'set_processes'")

    def add_transition_processes(self,*args,**kwargs):
        raise AttributeError("'SymbolicODEModel' object has no attribute 'add_transition_processes'")

    def add_fission_processes(self,*args,**kwargs):
        raise AttributeError("'SymbolicODEModel' object has no attribute 'add_fission_processes'")

    def add_fusion_processes(self,*args,**kwargs):
        raise AttributeError("'SymbolicODEModel' object has no attribute 'add_fusion_processes'")

    def add_transmission_processes(self,*args,**kwargs):
        raise AttributeError("'SymbolicODEModel' object has no attribute 'add_transmission_processes'")

    def add_quadratic_events(self,*args,**kwargs):
        raise AttributeError("'SymbolicODEModel' object has no attribute 'add_quadratic_events'")

    def add_linear_events(self,*args,**kwargs):
        raise AttributeError("'SymbolicODEModel' object has no attribute 'add_linear_events")


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

