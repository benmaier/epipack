
import numpy as np
from epipack import StochasticEpiModel

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

print("total event rate =", model.get_total_event_rate(), "; expected =",expected_rate)
print("total true event rate =", model.get_true_total_event_rate(), "; expected =",expected_true_rate)
