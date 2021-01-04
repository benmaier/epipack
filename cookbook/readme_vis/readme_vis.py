import epipack as epk
import epipack.vis as vis
import netwulf as nw

network, cfg, g = nw.load('MHRN.json')
N = len(network['nodes'])
links = [ (l['source'], l['target'], 1.0) for l in network['links'] ]

model = epk.StochasticEpiModel(["S","I","R"],N,links)\
            .set_link_transmission_processes([ ("I", "S", 1.0, "I", "I") ])\
            .set_node_transition_processes([ ("I", 1.0, "R") ])\
            .set_random_initial_conditions({ "S": N-5, "I": 5 })

vis.visualize(model, network, sampling_dt=0.1)
