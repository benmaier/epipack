
if __name__=="__main__":
    import networkx as nx
    from epipack import StochasticEpiModel


    k0 = 50
    N = int(1e4)

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
               well_mixed_mean_contact_number=k0,
               )\
           .set_link_transmission_processes(link_transmission)\
           .set_node_transition_processes(node_transition)\
           .set_conditional_link_transmission_processes(conditional_transmission)\
           .set_random_initial_conditions({
                                           'S': N-100,
                                           'I': 100
                                          })

    t, result = model.simulate(80)

    from epipack.plottools import plot

    ax = plot(t, result)
    ax.set_yscale('log')
    ax.set_ylim([1,N])
    ax.set_xlabel('time [d]')
    ax.legend()

    from bfmplot import pl
    ax.get_figure().savefig('SIARX_well_mixed.png',dpi=300)
    pl.show()
