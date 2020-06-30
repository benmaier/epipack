import numpy as np
from epipack import StochasticEpiModel

S, I, A, B, C0, C1, D, E, F = "S I A B C0 C1 D E F".split(" ")

rateA = 3.0
rateB = 2.0
rateE = 1.0

probAC0 = 0.2
probAC1 = 0.8

probBD = 0.5

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

print(model.node_transition_events)

counts = np.zeros(model.N_comp,dtype=int)

N_measurements = 10000

for meas in range(N_measurements):
    model.set_node_statuses(statuses)

    _ = model.simulate(1e9)
    for c in range(model.N_comp):

        counts[c] += np.count_nonzero(model.node_status == c)


from bfmplot import pl
x = np.arange(model.N_comp)
width = 0.4
pl.bar(x-width/2, counts, width)

expected_counts = np.zeros_like(counts)
expected_counts[model.get_compartment_id(A)] = N_measurements * rateA / (rateB + rateE + rateA)
expected_counts[model.get_compartment_id(B)] = N_measurements * rateB / (rateB + rateE + rateA)
expected_counts[model.get_compartment_id(E)] = N_measurements * rateE / (rateB + rateE + rateA)

expected_counts[model.get_compartment_id(C0)] = N_measurements * ((N-1)*rateA / (rateB + rateE + rateA) * probAC0)
expected_counts[model.get_compartment_id(C1)] = N_measurements * ((N-1)*rateA / (rateB + rateE + rateA) * probAC1)

expected_counts[model.get_compartment_id(D)] = N_measurements * ((N-1)*rateB / (rateB + rateE + rateA) * probBD)

expected_counts[model.get_compartment_id(F)] = (N-1) * expected_counts[model.get_compartment_id(E)]

expected_counts[model.get_compartment_id(S)] = N_measurements * N - expected_counts.sum()

pl.bar(x+width/2, expected_counts, width)

pl.xticks(x)
pl.gca().set_xticklabels(model.compartments)

pl.figure()

ndx = np.where(expected_counts==0)
counts[ndx] = 1
expected_counts[ndx] = 1

pl.plot(x, np.abs(1-counts/expected_counts))

from scipy.stats import entropy

_counts = np.delete(counts,1)
_exp_counts = np.delete(expected_counts,1)

print(entropy(_counts, _exp_counts))

pl.show()
