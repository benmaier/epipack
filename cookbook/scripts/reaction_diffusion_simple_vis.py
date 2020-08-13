from epipack import NumericMatrixBasedEpiModel, get_2D_lattice_links
from epipack.vis import visualize_reaction_diffusion, get_grid_layout
import numpy as np

from time import time

start = time()

N_side = 30
N = N_side**2
links = get_2D_lattice_links(N_side,periodic=False,diagonal_links=False)
network = get_grid_layout(N)

base_compartments = "SIR"

compartments = [ (node, C) for node in range(N) for C in base_compartments ]
model = NumericMatrixBasedEpiModel(compartments)

infection_rate = 2
recovery_rate = 1
mobility_rate = 0.1

quadratic_processes = []
linear_processes = []

print("defining processes")

for node in range(N):
    quadratic_processes.append(
            (  (node, "S"), (node, "I"), infection_rate, (node, "I"), (node, "I") ),
        )

    linear_processes.append(
              ( (node, "I"), recovery_rate, (node, "R") ) 
        )

for u, v, w in links:
    for C in base_compartments:

        linear_processes.extend([
                  ( (u, C), w*mobility_rate, (v, C) ),
                  ( (v, C), w*mobility_rate, (u, C) ),
            ])

print("set processes")

model.set_processes(quadratic_processes+linear_processes,allow_nonzero_column_sums=False)

initial_conditions = { ( node, "S" ): 1.0 for node in range(N) } 
initial_conditions[(0, "S")] = 0.99
initial_conditions[(0, "I")] = 0.01
model.set_initial_conditions(initial_conditions,allow_nonzero_column_sums=True)

t = np.linspace(0,20,100)

plot_node_indices = [ (node, "I") for node in range(N) ]

dt = 0.3

visualize_reaction_diffusion(model, network, dt, plot_node_indices, 
                        value_extent=[0,0.2],
                        config={
                                'draw_nodes_as_rectangles':True,
                                'draw_links':False
                                })

