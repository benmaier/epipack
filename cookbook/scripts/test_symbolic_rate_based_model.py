import sympy
import numpy as np
from epipack.symbolic_epi_models import SymbolicEventEpiModel
from bfmplot import pl

if __name__ == "__main__":
    S, I, t, eta, rho = sympy.symbols("S I t eta rho")
    N = 200
    model = SymbolicEventEpiModel([S,I],N,correct_for_dynamical_population_size=True)

    model.set_processes([
            (S, I, eta+sympy.cos(t/20*2*sympy.pi), I, I),            
            (None, rho*N, S),
            (I, rho, S),
            (S, rho, None),
            (I, rho, None),
        ])
    model.set_initial_conditions({
            S: 190,
            I: 10,
        })
    model.set_parameter_values({
            eta: 4,
            rho: 1,
        })

    def print_status():
        print(model.t0/50*100)

    t, result = model.simulate(50,sampling_dt=0.01,sampling_callback=print_status)    
    pl.plot(t, result[S],label='S')
    pl.plot(t, result[I],label='I')

    model.set_initial_conditions({
            S: 190,
            I: 10,
        })
    #tt = np.linspace(0,100,1000)
    tt = t
    result = model.integrate(tt)
    pl.plot(tt, result[S],label='S')
    pl.plot(tt, result[I],label='I')

    pl.show()

