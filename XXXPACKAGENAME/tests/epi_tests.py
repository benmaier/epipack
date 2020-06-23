import unittest

import numpy as np

from metapop.epi import (
            EpiModel,
            SISModel,
        )

class EpiTest(unittest.TestCase):

    def test_compartments(self):
        epi = EpiModel(list("SEIR"))
        assert(all([ i == epi.get_compartment_id(C) for i, C in enumerate("SEIR") ]))

    def test_mobility(self):
        epi = EpiModel(list("SEIR"))
        epi.set_compartment_mobility({
                "I": False,
                })
        mob = np.eye(4)
        mob[2,2] = 0
        assert(np.all(epi.mobility.toarray().flatten()==mob.flatten()))
        epi.set_compartment_mobility({
                "S": True,
                "E": True,
                "I": True,
                "R": True,
                })
        mob = np.eye(4)
        assert(np.all(epi.mobility.toarray().flatten()==mob.flatten()))

    def test_linear_rates(self):

        epi = EpiModel(list("SEIR"))
        epi.set_linear_processes([
                ("E", "I", 1.0),
                ("I", "R", 1.0),
                ])

        linear_rates = np.zeros((4,4))
        linear_rates[1,1] = -1
        linear_rates[2,1] = +1
        linear_rates[2,2] = -1
        linear_rates[3,2] = +1
        assert(np.all(epi.linear_rates.toarray().flatten()==linear_rates.flatten()))

    def test_quadratic_rates(self):

        epi = EpiModel(list("SEIR"))
        Q = [ np.zeros((4,4)) for C in epi.compartments ]
        Q[0][0,2] = -1
        Q[1][0,2] = +1
        epi.set_quadratic_processes([
                ("S", "I", "S", -1.0),
                ("S", "I", "E", +1.0),
                ])
        for iM, M in enumerate(epi.quadratic_rates):
            assert(np.all(M.toarray().flatten()==Q[iM].flatten()))

    def test_SIS(self):
        N = 100
        epi = SISModel(R0=2,recovery_rate=1,population_size=N)
        epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
        tt = np.linspace(0,100,2)
        result = epi.get_result_dict(tt,['S'])
        assert(np.isclose(result['S'][-1],N/2))


if __name__ == "__main__":

    T = Test()
    T.test_compartments()
    T.test_mobility()
    T.test_linear_rates()
    T.test_quadratic_rates()
    T.test_SIS()
