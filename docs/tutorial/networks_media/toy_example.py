from epipack import StochasticEpiModel
import numpy as np

if __name__ == "__main__":
    N = 3
    edge_weight_tuples = [
        (0, 1, 1/3),
        (0, 2, 2/3),
    ]
    directed = False

    S, I, R = list("SIR")

    model = StochasticEpiModel(
                    compartments=[S,I,R],
                    N=N,
                    edge_weight_tuples=edge_weight_tuples,
                    directed=directed,
                )

    infection_rate = 3.0
    model.set_link_transmission_processes([
            ('I', 'S', infection_rate, 'I', 'I'),
        ])

    recovery_rate = 2.0
    model.set_node_transition_processes([
            ('I', recovery_rate, 'R'),
        ])

    model.set_random_initial_conditions({S: N-1, I: 1})

    initial_node_statuses = np.array([
            model.get_compartment_id(I),
            model.get_compartment_id(S),
            model.get_compartment_id(S),
        ])
    model.set_node_statuses(initial_node_statuses)
    
    initial_node_statuses = np.zeros(N,dtype=int)
    initial_node_statuses[0] = 1
    model.set_node_statuses(initial_node_statuses)

    N_meas = 10000
    N_inf = 0
    N_inf_node_2 = 0
    for meas in range(N_meas):
        model.set_node_statuses(initial_node_statuses)
        model.simulate(tmax=1e300)
        if model.node_status[1] == model.get_compartment_id(R):
            N_inf += 1
        if model.node_status[2] == model.get_compartment_id(R):
            N_inf_node_2 += 1

    print("Node 1 has been infected in", N_inf/N_meas*100, "% of the measurements.")
    print("Node 2 has been infected in", N_inf_node_2/N_meas*100, "% of the measurements.")

    N_meas = 10000
    N_inf = 0
    reproduction_number = 0
    for meas in range(N_meas):
        model.set_node_statuses(initial_node_statuses)
        t, result = model.simulate(tmax=1e300)
        reproduction_number += result['R'][-1] -1 

    print("Node 0 has infected", reproduction_number/N_meas, "neighbors on average.")
    

