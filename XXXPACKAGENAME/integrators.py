"""
Contains integrator methods that will be used
to integrate ODEs or SDEs.
"""

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

if __name__ == "__main__":

    from epidemicmodeling import SISModel

    SIS = SISModel(R0=2.0,recovery_rate=1)
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
