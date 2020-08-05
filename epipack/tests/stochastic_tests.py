import unittest

import numpy as np
from scipy.stats import entropy

from epipack.stochastic_epi_models import (
            StochasticEpiModel,
        )

class StochasticEpiTest(unittest.TestCase):

    def test_compartments(self):
        epi = StochasticEpiModel(list("SEIR"),10)
        assert(all([ i == epi.get_compartment_id(C) for i, C in enumerate("SEIR") ]))

    def test_rate_evaluation(self):
        S, S0,S1, I, A, B, C0, C1, D, E, F = "S S0 S1 I A B C0 C1 D E F".split(" ")

        rateA = 3.0
        rateB = 2.0
        rateE = 1.0
        rateF = 5.0
        rateC = 10.0


        N = 6
        edges = [ (0, i, 1.0) for i in range(1,N) ]
        edges.extend([(N+1, 0, 1.0),(N,0,1.0),(N+1,1,1.0), (N+1,2,1.0)])

        model = StochasticEpiModel([S,S0,S1,I,A,B,C0,C1,D, E, F], N+2, edges)

        model.set_node_transition_processes([
                (S0, rateC, E),
                (I, rateF, F),
            ])

        model.set_link_transmission_processes([
                (S0, I, rateA, S0, A),
                (S0, I, rateB, S0, B),
                (S1, I, rateE, S1, E),
            ])

        statuses = np.zeros(N+2,dtype=int)
        statuses[0] = model.get_compartment_id(I)
        statuses[N] = model.get_compartment_id(S0)
        statuses[N+1] = model.get_compartment_id(S1)

        model.set_node_statuses(statuses)

        expected_rate = rateA + rateB + rateC + 3*rateE + rateF
        expected_true_rate = rateA + rateB + rateC + rateE + rateF

        assert(model.get_total_event_rate() == expected_rate)
        assert(model.get_true_total_event_rate() == expected_true_rate)

    def test_conditional_transitions(self):
        S, I, A, B, C0, C1, D, E, F = "S I A B C0 C1 D E F".split(" ")

        rateA = 3.0
        rateB = 2.0
        rateE = 1.0

        probAC0 = 0.2
        probAC1 = 0.8

        probBD = 0.2

        N = 6
        edges = [ (0, i, 1.0) for i in range(1,N) ]

        model = StochasticEpiModel([S,I,A,B,C0,C1,D, E, F], N, edges)

        model.set_node_transition_processes([
                (I, rateA, A),
                (I, rateB, B),
                (I, rateE, E),
            ])

        model.set_conditional_link_transmission_processes({
            (I, "->", A) : [
                    ( A, S, probAC0, A, C0),
                    ( A, S, probAC1, A, C1),
                ],
            (I, "->", B): [
                    ( B, S, probBD, B, D),
                ],
            (I, "->", E): [
                    ( E, S, "->", E, F),
                ],
            })

        statuses = np.zeros(N,dtype=int)
        statuses[0] = 1

        model.set_node_statuses(statuses)

        counts = np.zeros(model.N_comp,dtype=int)

        N_measurements = 10000

        for meas in range(N_measurements):
            model.set_node_statuses(statuses)

            _ = model.simulate(1e9)
            for c in range(model.N_comp):

                counts[c] += np.count_nonzero(model.node_status == c)

        expected_counts = np.zeros_like(counts)
        expected_counts[model.get_compartment_id(A)] = N_measurements * rateA / (rateB + rateE + rateA)
        expected_counts[model.get_compartment_id(B)] = N_measurements * rateB / (rateB + rateE + rateA)
        expected_counts[model.get_compartment_id(E)] = N_measurements * rateE / (rateB + rateE + rateA)

        expected_counts[model.get_compartment_id(C0)] = N_measurements * ((N-1)*rateA / (rateB + rateE + rateA) * probAC0)
        expected_counts[model.get_compartment_id(C1)] = N_measurements * ((N-1)*rateA / (rateB + rateE + rateA) * probAC1)

        expected_counts[model.get_compartment_id(D)] = N_measurements * ((N-1)*rateB / (rateB + rateE + rateA) * probBD)

        expected_counts[model.get_compartment_id(F)] = (N-1) * expected_counts[model.get_compartment_id(E)]

        expected_counts[model.get_compartment_id(S)] = N_measurements * N - expected_counts.sum()

        ndx = np.where(expected_counts==0)
        counts[ndx] = 1
        expected_counts[ndx] = 1

        _counts = np.delete(counts,1)
        _exp_counts = np.delete(expected_counts,1)
        ndx = np.where(_exp_counts == 0)
        assert(np.all(_counts[ndx]==0))
        ndx = np.where(_exp_counts > 0)
        rel_err = np.abs(1-_counts[ndx]/_exp_counts[ndx])
        assert(rel_err.mean() < 0.06)


    def test_conditional_transmissions(self):
        S, S0,S1, I, A, B, C0, C1, D, E, F = "S S0 S1 I A B C0 C1 D E F".split(" ")

        rateA = 3.0
        rateB = 2.0
        rateE = 1.0

        probAC0 = 0.2
        probAC1 = 0.8

        probBD = 0.2

        N = 6
        edges = [ (0, i, 1.0) for i in range(1,N) ]
        edges.extend([(N+1, 0, 1.0),(N,0,1.0)])

        model = StochasticEpiModel([S,S0,S1,I,A,B,C0,C1,D, E, F], N+2, edges)

        model.set_link_transmission_processes([
                (S0, I, rateA, S0, A),
                (S0, I, rateB, S0, B),
                (S1, I, rateE, S1, E),
            ])

        model.set_conditional_link_transmission_processes({
            (S0, I, "->", S0, A) : [
                    ( A, S, probAC0, A, C0),
                    ( A, S, probAC1, A, C1),
                ],
            (S0, I, "->", S0, B): [
                    ( B, S, probBD, B, D),
                ],
            (S1, I, "->", S1, E): [
                    ( E, S, "->", E, F),
                ],
            })

        statuses = np.zeros(N+2,dtype=int)
        statuses[0] = model.get_compartment_id(I)
        statuses[N+1] = model.get_compartment_id(S0)
        statuses[N] = model.get_compartment_id(S1)

        model.set_node_statuses(statuses)

        counts = np.zeros(model.N_comp,dtype=int)

        N_measurements = 10000

        for meas in range(N_measurements):
            model.set_node_statuses(statuses)

            _ = model.simulate(1e9)

            for c in range(model.N_comp):

                counts[c] += np.count_nonzero(model.node_status == c)


        expected_counts = np.zeros_like(counts)
        expected_counts[model.get_compartment_id(A)] = N_measurements * rateA / (rateB + rateE + rateA)
        expected_counts[model.get_compartment_id(B)] = N_measurements * rateB / (rateB + rateE + rateA)
        expected_counts[model.get_compartment_id(E)] = N_measurements * rateE / (rateB + rateE + rateA)

        expected_counts[model.get_compartment_id(C0)] = N_measurements * ((N-1)*rateA / (rateB + rateE + rateA) * probAC0)
        expected_counts[model.get_compartment_id(C1)] = N_measurements * ((N-1)*rateA / (rateB + rateE + rateA) * probAC1)

        expected_counts[model.get_compartment_id(D)] = N_measurements * ((N-1)*rateB / (rateB + rateE + rateA) * probBD)

        expected_counts[model.get_compartment_id(F)] = (N-1) * expected_counts[model.get_compartment_id(E)]

        expected_counts[model.get_compartment_id(S)] = N_measurements * N - expected_counts.sum()
        expected_counts[model.get_compartment_id(S0)] = N_measurements
        expected_counts[model.get_compartment_id(S1)] = N_measurements

        ndx = np.where(expected_counts==0)
        counts[ndx] = 1
        expected_counts[ndx] = 1

        _counts = np.delete(counts,1)
        _exp_counts = np.delete(expected_counts,1)
        
        ndx = np.where(_exp_counts == 0)
        assert(np.all(_counts[ndx]==0))
        ndx = np.where(_exp_counts > 0)
        rel_err = np.abs(1-_counts[ndx]/_exp_counts[ndx])
        assert(rel_err.mean() < 0.03)


if __name__ == "__main__":

    T = StochasticEpiTest()
    T.test_compartments()
    T.test_rate_evaluation()
    T.test_conditional_transitions()
    T.test_conditional_transmissions()