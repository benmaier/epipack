import unittest

import numpy as np
from scipy.optimize import root

from epipack.deterministic_epi_models import (
            DeterministicEpiModel,
            DeterministicSISModel,
            DeterministicSIModel,
            DeterministicSIRModel,
            DeterministicSIRSModel,
        )

class DeterministicEpiTest(unittest.TestCase):

    def test_compartments(self):
        epi = DeterministicEpiModel(list("SEIR"))
        assert(all([ i == epi.get_compartment_id(C) for i, C in enumerate("SEIR") ]))
        assert(epi.get_compartment_id("E") == 1)
        assert(epi.get_compartment(1) == "E")

    def test_linear_rates(self):

        epi = DeterministicEpiModel(list("SEIR"))
        epi.add_transition_processes([
                ("E", 1.0, "I"),
                ("I", 1.0, "R"),
                ])

        linear_rates = np.zeros((4,4))
        linear_rates[1,1] = -1
        linear_rates[2,1] = +1
        linear_rates[2,2] = -1
        linear_rates[3,2] = +1
        assert(np.all(epi.linear_rates.toarray().flatten()==linear_rates.flatten()))

    def test_adding_linear_rates(self):

        epi = DeterministicEpiModel(list("SEIR"))
        epi.set_processes([
                ("E", 1.0, "I"),
                ])

        epi.add_transition_processes([
                ("I", 1.0, "R"),
                ])

        linear_rates = np.zeros((4,4))
        linear_rates[1,1] = -1
        linear_rates[2,1] = +1
        linear_rates[2,2] = -1
        linear_rates[3,2] = +1
        assert(np.all(epi.linear_rates.toarray().flatten()==linear_rates.flatten()))

    def test_quadratic_processes(self):

        epi = DeterministicEpiModel(list("SEIR"))
        Q = [ np.zeros((4,4)) for C in epi.compartments ]
        Q[0][0,2] = -1
        Q[1][0,2] = +1
        epi.add_transmission_processes([
                ("S", "I",  1.0, "I", "E"),
                ])
        for iM, M in enumerate(epi.quadratic_rates):
            assert(np.all(M.toarray().flatten()==Q[iM].flatten()))

    def test_adding_quadratic_processes(self):

        epi = DeterministicEpiModel(list("SEIAR"))
        Q = [ np.zeros((5,5)) for C in epi.compartments ]
        Q[0][0,2] = -1
        Q[0][0,3] = -1
        Q[1][0,2] = +1
        Q[1][0,3] = +1
        epi.set_processes([
                ("S", "I",  1.0, "I", "E"),
                ])
        epi.add_transmission_processes([
                ("S", "A",  1.0, "A", "E"),
                ])
        for iM, M in enumerate(epi.quadratic_rates):
            assert(np.all(M.toarray().flatten()==Q[iM].flatten()))

    def test_SIS_with_simulation_restart_and_euler(self):
        N = 100
        epi = DeterministicSISModel(R0=2,recovery_rate=1,population_size=N)
        epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
        tt = np.linspace(0,100,2)
        result = epi.integrate(tt,['S'])
        assert(np.isclose(result['S'][-1],N/2))

        tt = np.linspace(0,100,1000)
        result = epi.integrate_and_return_by_index(tt,['S'],integrator='euler')
        assert(np.isclose(result[0,-1],N/2))

    def test_repeated_simulation(self):

        N = 100
        epi = DeterministicSISModel(R0=2,recovery_rate=1,population_size=N)
        epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
        tt = np.linspace(0,100,100)
        old_t = tt[0]

        for it, t in enumerate(tt[1:]):
            result = epi.integrate_and_return_by_index([old_t,t],integrator='euler',adopt_final_state=True)
            old_t = t

        assert(np.isclose(result[0,-1],N/2))

    def test_custom_models(self):

        S, I, R = list("SIR")

        eta = 1
        epi = DeterministicSIModel(eta)
        epi.set_initial_conditions({"S":0.99, "I":0.01})
        epi.integrate([0,1000],adopt_final_state=True)
        assert(np.isclose(epi.y0[0],0))


        
        eta = 2
        rho = 1
        epi = DeterministicSIRModel(eta,rho)
        S0 = 0.99
        epi.set_initial_conditions({S:S0, I:1-S0})
        R0 = eta/rho
        Rinf = lambda x: 1-x-S0*np.exp(-x*R0)
        res = epi.integrate([0,1000])

        theory = root(Rinf,0.5)
        assert(np.isclose(res[R][-1],theory.x[0]))

        epi = DeterministicSISModel(eta, rho, population_size=100)

        epi.set_initial_conditions({S: 99, I:1 })

        tt = np.linspace(0,1000,2)
        result = epi.integrate(tt)
        assert(np.isclose(result[S][-1],50))

        omega = 1
        epi = DeterministicSIRSModel(eta, rho, omega)

        epi.set_initial_conditions({S: 0.99, I:0.01 })

        tt = np.linspace(0,1000,2)
        result = epi.integrate(tt)
        assert(np.isclose(result[R][-1],(1-rho/eta)/(1+omega/rho)))


if __name__ == "__main__":

    T = DeterministicEpiTest()
    T.test_compartments()
    T.test_linear_rates()
    T.test_adding_linear_rates()
    T.test_quadratic_processes()
    T.test_adding_quadratic_processes()
    T.test_SIS_with_simulation_restart_and_euler()
    T.test_repeated_simulation()
    T.test_custom_models()
