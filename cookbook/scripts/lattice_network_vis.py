from epipack import visualize, get_grid_layout, get_2D_lattice_links


if __name__=="__main__":
    import netwulf as nw
    from epipack import StochasticEpiModel
    import networkx as nx

    N_side = 200
    N = N_side*N_side

    links = get_2D_lattice_links(N_side,periodic=True,diagonal_links=True)

    network = get_grid_layout(range(N),links,windowwidth=400)

    k0 = 2 * len(links) / N

    
    model = StochasticEpiModel(list("SIRXTQ"),N=N,
                                edge_weight_tuples=links,
                               )
    Reff = 3
    R0 = 4
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
                 ("X","I",0.0,"X","T"),
                 ("X","S",0.0,"X","Q"),
                 ],
        })
    model.set_random_initial_conditions({'I':20,'S':N-20})

    sampling_dt = 0.5

    visualize(model,network,sampling_dt,
              ignore_plot_compartments=['S','R'],
              quarantine_compartments=['X', 'T', 'Q'],
              config={'draw_nodes_as_rectanlges':True,'draw_links':False}
              )

