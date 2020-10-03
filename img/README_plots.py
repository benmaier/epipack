
def network_simulation():
    from epipack import StochasticEpiModel
    from epipack.matplotlib import plot
    import matplotlib.pyplot as pl
    import networkx as nx

    k0 = 50
    R0 = 2.5
    rho = 1
    eta = R0 * rho / k0
    omega = 1/14
    N = int(1e4)
    edges = [ (e[0], e[1], 1.0) for e in \
              nx.fast_gnp_random_graph(N,k0/(N-1)).edges() ]

    SIRS = StochasticEpiModel(
                compartments=list('SIR'),
                N=N,
                edge_weight_tuples=edges
                )\
            .set_link_transmission_processes([
                ('I', 'S', eta, 'I', 'I'),
            ])\
            .set_node_transition_processes([
                ('I', rho, 'R'),
                ('R', omega, 'S'),
            ])\
            .set_random_initial_conditions({
                                            'S': N-100,
                                            'I': 100
                                           })
    t_s, result_s = SIRS.simulate(40)

    ax = plot(t_s, result_s)
    ax.get_figure().savefig('network_simulation.png',dpi=300)


def network_simulation_visualization():
    from epipack import StochasticEpiModel
    from epipack.matplotlib import plot
    import matplotlib.pyplot as pl
    import networkx as nx

    k0 = 50
    R0 = 2.5
    rho = 1
    eta = R0 * rho / k0
    omega = 1/14
    N = int(1e4)
    edges = [ (e[0], e[1], 1.0) for e in \
              nx.fast_gnp_random_graph(N,k0/(N-1)).edges() ]

    SIRS = StochasticEpiModel(
                compartments=list('SIR'),
                N=N,
                edge_weight_tuples=edges
                )\
            .set_link_transmission_processes([
                ('I', 'S', eta, 'I', 'I'),
            ])\
            .set_node_transition_processes([
                ('I', rho, 'R'),
                ('R', omega, 'S'),
            ])\
            .set_random_initial_conditions({
                                            'S': N-100,
                                            'I': 100
                                           })

    from epipack.vis import visualize
    from epipack.networks import get_random_layout
    layouted_network = get_random_layout(N, edges)
    visualize(SIRS, layouted_network, sampling_dt=0.1, config={'draw_links': False})

def MHRN_network_vis():
    import epipack as epk
    from epipack.vis import visualize
    import netwulf as nw

    network, _, __ = nw.load('../cookbook/readme_vis/MHRN.json')
    N = len(network['nodes'])
    links = [ (l['source'], l['target'], 1.0) for l in network['links'] ]

    model = epk.StochasticEpiModel(["S","I","R"],N,links)\
                .set_link_transmission_processes([ ("I", "S", 1.0, "I", "I") ])\
                .set_node_transition_processes([ ("I", 1.0, "R") ])\
                .set_random_initial_conditions({ "S": N-5, "I": 5 })

    visualize(model, network, sampling_dt=0.1)

def default_numeric_models():
    from epipack import EpiModel
    import matplotlib.pyplot as pl
    from epipack.matplotlib import plot
    import numpy as np

    S, I, R = list("SIR")
    N = 1000

    SIRS = EpiModel([S,I,R],N)\
        .set_processes([
            #### transmission process ####
            # S + I (eta=2.5/d)-> I + I
            (S, I, 2.5, I, I),

            #### transition processes ####
            # I (rho=1/d)-> R
            # R (omega=1/14d)-> S
            (I, 1, R),
            (R, 1/14, S),
        ])\
        .set_initial_conditions({S:N-10, I:10})

    t = np.linspace(0,40,1000)
    result_int = SIRS.integrate(t)
    t_sim, result_sim = SIRS.simulate(t[-1])

    ax = plot(t_sim, result_sim)
    ax = plot(t, result_int,ax=ax)
    ax.get_figure().savefig('numeric_model.png',dpi=300)

def varying_rate_numeric_models():

    import numpy as np
    from epipack import SISModel
    from epipack.matplotlib import plot

    N = 100
    recovery_rate = 1.0

    def infection_rate(t, y, *args, **kwargs):
        return 3 + np.sin(2*np.pi*t/100)

    SIS = SISModel(
                infection_rate=infection_rate,
                recovery_rate=recovery_rate,
                initial_population_size=N
                )\
            .set_initial_conditions({
                'S': 90,
                'I': 10,
            })

    t = np.arange(200)
    result_int = SIS.integrate(t)
    t_sim, result_sim = SIS.simulate(199)

    ax = plot(t_sim, result_sim)
    ax = plot(t, result_int,ax=ax)
    ax.get_figure().savefig('numeric_model_time_varying_rate.png',dpi=300)

def symbolic_simulation_temporally_forced():
    from epipack import SymbolicEpiModel
    import sympy as sy
    from epipack.matplotlib import plot
    import numpy as np

    S, I, R, eta, rho, omega, t, T = \
            sy.symbols("S I R eta rho omega t T")

    N = 1000
    SIRS = SymbolicEpiModel([S,I,R],N)\
        .set_processes([
            (S, I, 3+sy.cos(2*sy.pi*t/T), I, I),
            (I, rho, R),
            (R, omega, S),
        ])

    SIRS.set_parameter_values({
        rho : 1,
        omega : 1/14,
        T : 100,
    })
    SIRS.set_initial_conditions({S:N-100, I:100})
    _t = np.linspace(0,150,1000)
    result = SIRS.integrate(_t)
    t_sim, result_sim = SIRS.simulate(max(_t))

    ax = plot(_t, result)
    plot(t_sim, result_sim, ax=ax)
    ax.get_figure().savefig('symbolic_model_time_varying_rate.png',dpi=300)


if  __name__=="__main__":
    #default_numeric_models()
    #varying_rate_numeric_models()
    symbolic_simulation_temporally_forced()
    #network_simulation()
    #network_simulation_visualization()
