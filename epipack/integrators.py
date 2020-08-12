"""
Contains integrator methods that will be used
to integrate ODEs or SDEs.
"""

import warnings

import numpy as np
from scipy.integrate import ode

def integrate_dopri5(dydt, t, y0, *args):
    """
    Integrate an ODE system with the Runge-Kutte Dormand Prince
    method (with step-size control).

    Parameters
    ----------
    dydt : function
        A function returning the momenta of the ODE system.
    t : numpy.ndarray of float
        Array of time points at which the functions should
        be evaluated.
    y0 : numpy.ndarray
        Initial conditions
    *args : :obj:`list`
        List of parameters that will be passed to the
        momenta function.
    """

    # get copy
    y0 = np.array(y0,dtype=float)
    t = np.array(t,dtype=float)

    t0 = t[0]
    t = t[1:]

    # initiate integrator
    r = ode(dydt)
    r.set_integrator('dopri5')
    r.set_initial_value(y0,t0)
    r.set_f_params(*args)
    result = np.zeros((len(y0),len(t)+1))
    result[:,0] = y0


    # loop through all demanded time points
    for it, t_ in enumerate(t):

            # get result of ODE integration
            y = r.integrate(t_)

            # write result to result vector
            result[:,it+1] = y

    return result

def integrate_euler(dydt, t, y0, *args):
    """
    Integrate an ODE system with Euler's method.

    Parameters
    ----------
    dydt : function
        A function returning the momenta of the ODE system.
    t : numpy.ndarray of float
        Array of time points at which the functions should
        be evaluated.
    y0 : numpy.ndarray
        Initial conditions
    *args : :obj:`list`
        List of parameters that will be passed to the
        momenta function.
    """

    # get copy
    y0 = np.array(y0,dtype=float)
    t = np.array(t,dtype=float)

    t0 = t[0]


    # initiate integrator
    result = np.zeros((len(y0),len(t)+1))
    result[:,0] = y0
    old_t = t0


    # loop through all demanded time points
    for it, t_ in enumerate(t):

            # get result of ODE integration
            dt = t_ - old_t
            y = result[:,it] + dt*dydt(old_t, result[:,it], *args) 

            # write result to result vector
            result[:,it+1] = y

            old_t = t_

    return result

class IntegrationMixin():

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

        dydt = self.get_numerical_dydt()

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

    def integrate(self,
                  time_points,
                  return_compartments=None,
                  *args,
                  **kwargs,
                  ):
        r"""
        Returns values of the given compartments at the demanded
        time points (as a dictionary).

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
            as new initial conditions.
        """
        if return_compartments is None:
            return_compartments = self.compartments

        result = self.integrate_and_return_by_index(time_points, return_compartments,*args,**kwargs)

        result_dict = {}
        for icomp, compartment in enumerate(return_compartments):
            result_dict[compartment] = result[icomp,:]

        return result_dict

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

        if np.abs(total-self.initial_population_size)/self.initial_population_size > 1e-14 and not allow_nonzero_column_sums:
            warnings.warn('Sum of initial conditions does not equal unity.')

        return self

if __name__ == "__main__":     # pragma: no cover

    from epipack import DeterministicSISModel

    SIS = DeterministicSISModel(R0=2.0,recovery_rate=1)
    SIS.set_initial_conditions({
                    'S': 0.99,
                    'I': 0.01,
                })


    tt = np.linspace(0,10,100)
    result = integrate_dopri5(SIS.dydt, tt, SIS.y0)

    import matplotlib.pyplot as pl

    pl.plot(tt, result[0,:])
    pl.plot(tt, result[1,:])
    
    pl.show()
