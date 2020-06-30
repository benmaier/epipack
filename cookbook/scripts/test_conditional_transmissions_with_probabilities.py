import numpy as np
from epipack import StochasticEpiModel
from tqdm import tqdm

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
print(edges)

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

print(I, model.get_compartment_id(I))
statuses = np.zeros(N+2,dtype=int)
statuses[0] = model.get_compartment_id(I)
statuses[N+1] = model.get_compartment_id(S0)
statuses[N] = model.get_compartment_id(S1)

model.set_node_statuses(statuses)

#import networkx as nx
#import netwulf as nw
#G = nx.Graph()
#G.add_edges_from([e[:2] for e in edges])
#for n, col in enumerate(statuses):
#    G.nodes[n]['group'] = col
#nw.visualize(G)


print(statuses)
print(model.link_transmission_events)

counts = np.zeros(model.N_comp,dtype=int)

N_measurements = 10000

for meas in tqdm(range(N_measurements)):
    model.set_node_statuses(statuses)

    #import sys
    #sys.exit(1)
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
expected_counts[model.get_compartment_id(S0)] = N_measurements
expected_counts[model.get_compartment_id(S1)] = N_measurements

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
