"""
Provides an API to define deterministic epidemiological models in terms of sympy symbolic expressions.
"""

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

from epipack.deterministic_epi_models import DeterministicEpiModel

class SymbolicRateBasedMixin(DeterministicEpiModel):
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

    def __init__(self,compartments,population_size=1):
        """
        """
        DeterministicEpiModel.__init__(self, compartments, population_size)


        self.t = sympy.symbols("t")
        if self.t in self.compartments:
            raise ValueError("Don't use `t` as a compartment symbol, as it is reserved for time.")

        self.has_functional_rates = False

        self.birth_rates = sympy.zeros(self.N_comp,1)
        self.linear_rates = sympy.zeros(self.N_comp, self.N_comp)
        self.quadratic_rates = [ sympy.zeros(self.N_comp, self.N_comp)\
                                 for c in range(self.N_comp) ]
        self.parameter_values = {}

    def _check_rate_for_functional_dependency(self,rate):
        try:
            self.has_functional_rates |= any([ compartment in rate.free_symbols for compartment in self.compartments])
        except AttributeError as e:
            return

    def dydt(self):
        """
        Obtain the equations of motion for this model in form of a matrix.
        """

        y = sympy.Matrix(self.compartments)
        
        ynew = self.linear_rates * y + self.birth_rates
        for c in self.affected_by_quadratic_process:
            ynew[c] += (y.T * self.quadratic_rates[c] * y)[0,0] / self.population_size

        return ynew
            
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
        
    def integrate_and_return_by_index(self,
                                      time_points,
                                      return_compartments=None,
                                      integrator='dopri5',
                                      adopt_final_state=False,
                                      ):
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

        if integrator == 'dopri5':
            result = integrate_dopri5(dydt, time_points, self.y0)
        else:
            result = integrate_euler(dydt, time_points, self.y0)

        if adopt_final_state:
            self.y0 = result[:,-1]

        if return_compartments is not None:
            ndx = [self.get_compartment_id(C) for C in return_compartments]
            result = result[ndx,:]

        return result

