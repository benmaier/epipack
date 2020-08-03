import numpy as np
from epipack.stochastic_epi_models import StochasticEpiModel
from SIRXQ import mean_field_SIRX_tracing
import networkx as nx

from time import time

if __name__=="__main__":

    N = 1000
    I0 = 50
    k0 = 100
    R0 = 2.5
    Q0 = 0.5
    waning_quarantine_rate = omega = 1/14
    recovery_rate = rho = 1/2
    quarantine_rate = kappa = rho * Q0 / (1-Q0)
    infection_rate = eta = R0 * (rho) / k0

    p = k0/(N-1)
    G = nx.fast_gnp_random_graph(N, p)
    edges = [ (e[0], e[1], 1.0) for e in G.edges() ]

    #model = StochasticEpiModel(list("SIXRQ"),N,edge_weight_tuples=edges)
    model = StochasticEpiModel(list("SIXRQ"),N,well_mixed_mean_contact_number=k0) 

    model.set_node_transition_processes([
            ("I",rho,"R"),
            ("I",kappa,"X"),
            ("Q",omega,"S"),
        ])

    model.set_link_transmission_processes([
            ("I","S",eta,"I","I"),
        ])

    model.set_conditional_link_transmission_processes({
            ( "I", "->", "X" ) : [ 
                        ("X","S","->","X","Q"), 
                        ("X","I","->","X","X"), 
                    ]
        })

    print(model.conditional_link_transmission_events)
    model.set_random_initial_conditions({"S": N-I0, "I": I0})

    print(model.node_and_link_events)
    start = time()
    t, result = model.simulate(300,sampling_dt=1)
    end = time()

    print("simulation took", end-start,"seconds")

    from bfmplot import pl

    for comp, series in result.items():
        pl.plot(t, series, label=comp)

    kw = {
             'R0': R0,
             'Q0': Q0,
             'k0': k0,
             'I0': I0/N,
             'waning_quarantine_rate': waning_quarantine_rate,
             'recovery_rate': recovery_rate,
             'infection_rate': infection_rate,
             'app_participation_ratio': 1.0,
             't': t,
         }

    t, result = mean_field_SIRX_tracing(kw)
             
    for comp, series in result.items():
        pl.plot(t, series*N, label=comp)


    pl.legend()
    pl.show()

