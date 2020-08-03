from epipack.vis import visualize, get_grid_layout
from epipack import StochasticSISModel


if __name__=="__main__":

    N = 100 * 100
    network = get_grid_layout(N)
    edge_list = [ ( link['source'], link['target'], 1.0 ) for link in network['links'] ]

    R0 = 3
    recovery_rate = 1/8
    model = StochasticSISModel(N,R0,recovery_rate)
    model.set_random_initial_conditions({'I':20,'S':N-20})

    sampling_dt = 0.5

    visualize(model,network,sampling_dt,
              config={'draw_nodes_as_rectangles':True}
              )

