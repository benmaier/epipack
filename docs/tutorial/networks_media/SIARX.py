
if __name__=="__main__":
    import networkx as nx
    from epipack import StochasticEpiModel, EpiModel
    import numpy as np


    k0 = 50
    N = int(1e4)
    edges = [ (e[0], e[1], 1.0) for e in \
              nx.fast_gnp_random_graph(N,k0/(N-1)).edges() ]

    recovery_rate = 1/5
    quarantine_rate = 1/5

    Reff = 2.0 
    infection_rate = Reff * (recovery_rate + quarantine_rate) / k0

    p = 0.25

    compartments = list("SIARX")

    node_transition = [
        ('I', recovery_rate, 'R'),
        ('A', recovery_rate, 'R'),
        ('I', quarantine_rate, 'X'),
    ]

    link_transmission = [
        ('I', 'S', infection_rate*0.7, 'I', 'I'),
        ('I', 'S', infection_rate*0.3, 'I', 'A'),
        ('A', 'S', 0.5*infection_rate*0.7, 'A', 'I'),
        ('A', 'S', 0.5*infection_rate*0.3, 'A', 'A'),
    ]

    conditional_transmission = {
        ('I', '->', 'X') : [
                ('X', 'I', p, 'X', 'X'),
            ]
    }

    model = StochasticEpiModel(
               compartments=compartments,
               N=N,
               edge_weight_tuples=edges
               )\
           .set_link_transmission_processes(link_transmission)\
           .set_node_transition_processes(node_transition)\
           .set_conditional_link_transmission_processes(conditional_transmission)\
           .set_random_initial_conditions({
                                           'S': N-100,
                                           'I': 100
                                          })

    t, result = model.simulate(100)

    from epipack.plottools import plot
    from bfmplot import pl

    ax = plot(t, result)
    ax.set_yscale('log')
    ax.set_ylim([1,N])
    ax.set_ylim([1,N])
    ax.set_xlabel('time [d]')
    ax.legend()

    ax.get_figure().savefig('SIARX_network.png',dpi=300)

    link_transmission = [
        ('I', 'S', k0*infection_rate*0.7, 'I', 'I'),
        ('I', 'S', k0*infection_rate*0.3, 'I', 'A'),
        ('A', 'S', k0*0.5*infection_rate*0.7, 'A', 'I'),
        ('A', 'S', k0*0.5*infection_rate*0.3, 'A', 'A'),
        ('I', 'I', quarantine_rate*k0*p, 'I', 'X'),
    ]

    model = EpiModel(
               compartments=compartments,
               initial_population_size=N,
               )\
           .set_processes(node_transition + link_transmission)\
           .set_initial_conditions({
                                       'S': N-100,
                                       'I': 100
                                   })
    t = np.linspace(0,t[-1],1000)
    result = model.integrate(t)

    ax = plot(t, result)
    ax.set_yscale('log')
    ax.set_ylim([1,N])
    ax.set_ylim([1,N])
    ax.set_xlabel('time [d]')
    ax.legend()

    ax.get_figure().savefig('SIARX_deterministics.png',dpi=300)

    pl.show()
