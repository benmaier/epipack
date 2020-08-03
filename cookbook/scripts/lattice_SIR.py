from epipack.vis import visualize, get_grid_layout
from epipack import StochasticSIRModel, get_2D_lattice_links

# define links and network layout
N_side = 100
N = N_side**2
links = get_2D_lattice_links(N_side, periodic=True, diagonal_links=True)
network = get_grid_layout(N)

# define model
R0 = 3; recovery_rate = 1/8
model = StochasticSIRModel(N,R0,recovery_rate,
                           edge_weight_tuples=links)
model.set_random_initial_conditions({'I':20,'S':N-20})

sampling_dt = 1

visualize(model,network,sampling_dt,
        config={
             'draw_nodes_as_rectangles':True,
             'draw_links':False,
           }
      )
