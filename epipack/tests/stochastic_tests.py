import unittest

import numpy as np
from scipy.stats import entropy

from epipack.stochastic_epi_models import (
            StochasticEpiModel,
            StochasticSIModel,
            StochasticSIRModel,
            StochasticSISModel,
        )

class StochasticEpiTest(unittest.TestCase):

    def test_compartments(self):
        epi = StochasticEpiModel(list("SEIR"),10)
        assert(all([ i == epi.get_compartment_id(C) for i, C in enumerate("SEIR") ]))
        assert(epi.get_compartment_id("E") == 1)
        assert(epi.get_compartment(1) == "E")

    def test_mean_contact_number(self):
        self.assertRaises(TypeError,StochasticEpiModel,list("SEIR"),10,well_mixed_mean_contact_number=0.5)

    def test_errors(self):
        epi = StochasticEpiModel(list("SEIR"),10)
        self.assertRaises(ValueError,epi.set_link_transmission_processes,[("S","I",1.0,"I","I")])

        # error: first compartment has to be infecting compartment
        self.assertRaises(ValueError,epi.set_conditional_link_transmission_processes,
                    { ("S", "I", "->", "I", "I") :
                        [
                            ("I", "R", "->", "I", "I"),
                        ]
                    }
                )

        # error: 4-length tuples are invalid processes
        self.assertRaises(ValueError,epi.set_conditional_link_transmission_processes,
                    { ("S", "I", "I", "I") :
                        [
                            ("I", "R", "->", "I", "I"),
                        ]
                    }
                )

        # error triggered events can only be of type transmission
        self.assertRaises(ValueError,epi.set_conditional_link_transmission_processes,
                    { ("I", "S", "->", "I", "I") :
                        [
                            ("R", "->", "I"),
                        ]
                    }
                )

        # error: target compartment of triggering event has to be first source compartment
        # of triggered event
        self.assertRaises(ValueError,epi.set_conditional_link_transmission_processes,
                    { ("I", "S", "->", "I", "I") :
                        [
                            ("R", "I", "->", "I", "I"),
                        ]
                    }
                )
            
        # error: "The source (infecting) compartment", coupling0, "must be equal to the affected (target) compartment of",

        #"the triggering event", triggering_event, " but is not."
        self.assertRaises(ValueError,epi.set_conditional_link_transmission_processes,
                    { ("I", "S", "->", "I", "I") :
                        [
                            ("R", "S", "->", "R", "I"),
                        ]
                    }
                )

        #probabilities sum to value large than one
        self.assertRaises(ValueError,epi.set_conditional_link_transmission_processes,
                    { ("I", "S", "->", "I", "I") :
                        [
                            ("I", "R", 0.6, "I", "E"),
                            ("I", "R", 0.6, "I", "S"),
                        ]
                    }
                )

        # error: double entry
        self.assertRaises(ValueError,epi.set_random_initial_conditions,
                    [ ("S",2),("S",1) ]
                    )

        # error: sum of initial conditions
        self.assertRaises(ValueError,epi.set_random_initial_conditions,
                    [ ("S",5),("I",1) ]
                    )

        # error: not the right amount of node statuses
        self.assertRaises(ValueError,epi.set_node_statuses,[0,1])

        self.assertRaises(ValueError,epi.simulate,10,sampling_callback=lambda x: x)

    def test_random_initial_conditions(self):
        epi = StochasticEpiModel(list("SEIR"),10)
        epi.set_node_transition_processes([("I",1,"R")])
        epi.set_random_initial_conditions({
                "S": 4,
                "I": 6,
                "R": 0,
            })

        assert(np.count_nonzero(epi.node_status==0) == 4)
        assert(np.count_nonzero(epi.node_status==2) == 6)
        assert(np.count_nonzero(epi.node_status==3) == 0)



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

    def test_sampling_callback(self):
        epi = StochasticSIModel(100,infection_rate=5.0)
        epi.set_random_initial_conditions({"S":90,"I":10})
        self.assertRaises(ValueError,epi.simulate,1,sampling_callback=lambda x: x)

        i = 0
        samples = []
        def sampled():
            samples.append(epi.y0[0])

        t, res = epi.simulate(10,sampling_dt=0.1,sampling_callback=sampled)

        assert(all([a==b for a, b in zip(res['S'], samples)]))


    def test_total_event_rate_well_mixed(self):
        N = 4
        I0 = 2
        S0 = N-I0
        k0 = 1
        R0 = 10.0
        rho = 1.0
        eta = R0*rho
        epi = StochasticSIRModel(N,R0=R0,recovery_rate=rho,well_mixed_mean_contact_number=k0)
        epi.set_random_initial_conditions({"S":S0,"I":I0})
        SI_rate = I0*eta*S0/(N-1)
        I_rate = rho*I0
        print(epi.get_true_total_event_rate(), I_rate + SI_rate)
        print(epi.get_total_event_rate(), I_rate + SI_rate)

        assert(np.isclose(epi.get_true_total_event_rate(), I_rate + SI_rate))
        assert(epi.get_true_total_event_rate() < epi.get_total_event_rate())

    
        # This is for testing whether the true event rate should be corrected with N-1 and not with N
        N_sample =  3000

        count_SI = 0
        taus = []
        SI_taus = []
        for sample in range(N_sample):
            epi.set_random_initial_conditions({"S":S0,"I":I0})
            t, res = epi.simulate(1)
            if res['I'][1] > I0:
                count_SI += 1
                SI_taus.append(t[1]-t[0])
            taus.append(t[1]-t[0])

        SI_rate_2 = I0*eta*S0/N
        I_rate = rho*I0
        #print(count_SI/N_sample, SI_rate/(I_rate+SI_rate), )
        #print(count_SI/N_sample, SI_rate_2/(I_rate+SI_rate_2), )

        from matplotlib import pyplot as pl
        pl.figure()
        counts, bins = np.histogram(taus,bins=100,density=True)
        x = 0.5*(bins[1:]+bins[:-1])
        #pl.hist(taus,bins=50,density=True)
        L = I_rate + SI_rate
        L2 = I_rate + SI_rate_2
        #pl.plot(x,L*np.exp(-L*x))
        #pl.plot(x,L2*np.exp(-L2*x))

        #pl.figure()
        counts, bins = np.histogram(SI_taus,bins=100,density=True)
        x = 0.5*(bins[1:]+bins[:-1])
        #pl.hist(SI_taus,bins=50,density=True)
        L = SI_rate
        L2 = SI_rate_2
        #pl.plot(x,L*np.exp(-L*x))
        #pl.plot(x,L2*np.exp(-L2*x))

        #pl.show()


if __name__ == "__main__":

    T = StochasticEpiTest()
    T.test_total_event_rate_well_mixed()
    T.test_sampling_callback()
    T.test_compartments()
    T.test_mean_contact_number()
    T.test_errors()
    T.test_random_initial_conditions()
    T.test_rate_evaluation()
    T.test_conditional_transitions()
    T.test_conditional_transmissions()
