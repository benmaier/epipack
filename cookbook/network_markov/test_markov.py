import numpy as np
import epipack as epk
import networkx as nx

N = 3
R, E, I, S = comps = list("REIS")
iR, iE, iI, iS = range(len(comps))

S, E, I, R = comps = list("SEIR")
iS, iE, iI, iR = range(len(comps))

edges = [
            (0, 1, 1.0),
            (1, 2, 3.0),
            (2, 0, 2.0),
        ]
directed = True

alpha = 1.0
beta = 1.0
gamma = 0.1

node_statuses = [iS if i>0 else iI for i in range(N) ]

transmission = [
            (I, S, alpha, I, E)
        ]
transition = [
            (E, gamma, I),
            (I, beta, R),
        ]

gilles = ( epk.StochasticEpiModel(comps, N, edges, directed=directed)
              .set_node_transition_processes(transition)
              .set_link_transmission_processes(transmission)
              .set_node_statuses(node_statuses)
          )

print(gilles.node_status)
markov = ( epk.NetworkMarkovEpiModel(comps, N, edges, directed=directed)
              .set_node_transition_processes(transition)
              .set_link_transmission_processes(transmission)
              .set_node_statuses(node_statuses)
          )


print(markov.adjacency_matrix.toarray())

for comp, Q in zip(comps, markov.matrix_model.quadratic_rates):
    print(comp,'\n',Q.toarray())

print(markov.y0)
print(markov.y0.reshape(len(comps),N))

net_comps = [
             (C, i) for C in comps for i in range(N)
            ]

net_transmission = []
net_transition = []

for src, trg, w in edges:
    net_transmission.append(
            ( (S,trg), (I,src), w*alpha, (I,src), (E,trg) )
        )
    if not directed:
        net_transmission.append(
                ( (S,src), (I,trg), w*alpha, (I,trg), (E,src) )
            )

init = {}
for node in range(N):
    net_transition.extend([
            ( (E,node), gamma, (I, node)),
            ( (I,node), beta, (R,node)),
        ])

    init[comps[node_statuses[node]], node] = 1.0

print(init)

markov2 = ( epk.MatrixEpiModel(net_comps)
               .add_transition_processes(net_transition)
               .add_transmission_processes(net_transmission)
               .set_initial_conditions(init)
           )

tt = np.linspace(0,20,21)

result = markov.integrate(tt)
result2 = markov2.integrate(tt)


from epipack.plottools import plot, pl

fig, ax = pl.subplots(1,2,figsize=(8,4))
plot(tt, result,ax=ax[0])
ax[0].legend()
plot(tt, result2,ax=ax[1])
ax[1].legend()

ax2 = plot(tt, markov.collapse_result_to_status_counts(result))
ax2.legend()



pl.show()
