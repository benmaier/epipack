from epipack.vis import visualize


if __name__=="__main__":
    import netwulf as nw
    from epipack import StochasticEpiModel
    import networkx as nx

    network, config, _  = nw.load('random_network.json')

    N = len(network['nodes'])

    edge_list = [ ( link['source'], link['target'], 1.0 ) for link in network['links'] ]
    k0 = 2*len(edge_list)/len(network['nodes'])

    
    model = StochasticEpiModel(list("SIRXTQ"),N=len(network['nodes']),
                               edge_weight_tuples=edge_list
                               )
    Reff = 3
    R0 = 3
    recovery_rate = 1/8
    quarantine_rate = 1/32
    tracing_rate = 1/2
    waning_immunity_rate = 1/14
    infection_rate = Reff * (recovery_rate+quarantine_rate) / k0
    infection_rate = R0 * (recovery_rate) / k0
    model.set_node_transition_processes([
            ("I",recovery_rate,"R"),
            ("I",quarantine_rate,"T"),
            ("T",tracing_rate,"X"),
            ("Q",waning_immunity_rate,"S"),
            ("X",recovery_rate,"R"),
            ])
    model.set_link_transmission_processes([("I","S",infection_rate,"I","I")])
    model.set_conditional_link_transmission_processes({
        ("T", "->", "X") : [
                 ("X","I",0.4,"X","T"),
                 ("X","S",0.4,"X","Q"),
                 ],
        })
    model.set_random_initial_conditions({'I':20,'S':N-20})

    sampling_dt = 0.5

    visualize(model,
              network,
              sampling_dt,
              ignore_plot_compartments=['S'],
              quarantine_compartments=['X', 'T', 'Q'],
              config={'draw_links':True}
              )

