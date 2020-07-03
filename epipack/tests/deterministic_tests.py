import unittest

import numpy as np

from epipack.deterministic_epi_models import (
            DeterministicEpiModel,
            DeterministicSISModel,
        )

class EpiTest(unittest.TestCase):

    def test_compartments(self):
        epi = DeterministicEpiModel(list("SEIR"))
        assert(all([ i == epi.get_compartment_id(C) for i, C in enumerate("SEIR") ]))

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

    def test_quadratic_rates(self):

        epi = DeterministicEpiModel(list("SEIR"))
        Q = [ np.zeros((4,4)) for C in epi.compartments ]
        Q[0][0,2] = -1
        Q[1][0,2] = +1
        epi.add_transmission_processes([
                ("S", "I",  1.0, "I", "E"),
                ])
        for iM, M in enumerate(epi.quadratic_rates):
            assert(np.all(M.toarray().flatten()==Q[iM].flatten()))

    def test_SIS(self):
        N = 100
        epi = DeterministicSISModel(R0=2,recovery_rate=1,population_size=N)
        epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
        tt = np.linspace(0,100,2)
        result = epi.integrate(tt,['S'])
        assert(np.isclose(result['S'][-1],N/2))


if __name__ == "__main__":

    T = EpiTest()
    T.test_compartments()
    T.test_linear_rates()
    T.test_quadratic_rates()
    T.test_SIS()
