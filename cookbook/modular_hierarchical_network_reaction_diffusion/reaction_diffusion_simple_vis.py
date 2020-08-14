from epipack import MatrixEpiModel, get_2D_lattice_links
from epipack.vis import visualize_reaction_diffusion, get_grid_layout
import numpy as np

import netwulf as nw

from time import time

start = time()

network, config, _ = nw.load('MHRN.json')

# get the network properties
N = len(network['nodes'])
links = [ ( link['source'], link['target'], 1.0 ) for link in network['links'] ]


base_compartments = "SIR"

compartments = [ (node, C) for node in range(N) for C in base_compartments ]
model = MatrixEpiModel(compartments)

infection_rate = 3
recovery_rate = 1
mobility_rate = 0.05

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

model.set_processes(quadratic_processes+linear_processes,allow_nonzero_column_sums=True)

initial_conditions = { ( node, "S" ): 1.0 for node in range(N) } 
initial_conditions[(0, "S")] = 0.8
initial_conditions[(0, "I")] = 0.2
model.set_initial_conditions(initial_conditions,allow_nonzero_column_sums=True)

t = np.linspace(0,20,100)

node_compartments = [ (node, "I" ) for node in range(N) ]

dt = 0.04
visualize_reaction_diffusion(model, network, dt, node_compartments, value_extent=[0,0.3])

