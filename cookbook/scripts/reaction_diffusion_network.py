from epipack import DeterministicEpiModel, get_2D_lattice_links
from epipack.vis import visualize_reaction_diffusion, get_random_layout
import numpy as np

from time import time

start = time()

N = 300
k = 4
links = []
for i in range(N):
    neighs = np.random.randint(0,N-1,size=(k,),dtype=int)
    neighs[neighs>=i] += 1
    for neigh in neighs:
        links.append((i,int(neigh),1.0))

network = get_random_layout(N,links,windowwidth=500)

# get the network properties
N = len(network['nodes'])
links = [ ( link['source'], link['target'], 1.0 ) for link in network['links'] ]


base_compartments = "SIR"

compartments = [ (node, C) for node in range(N) for C in base_compartments ]
model = DeterministicEpiModel(compartments)

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

model.set_processes(quadratic_processes+linear_processes,allow_nonzero_column_sums=False)

initial_conditions = { ( node, "S" ): 1.0 for node in range(N) } 
initial_conditions[(0, "S")] = 0.99
initial_conditions[(0, "I")] = 0.01
model.set_initial_conditions(initial_conditions,allow_nonzero_column_sums=True)

t = np.linspace(0,20,100)

plot_node_indices = [ (node, model.get_compartment_id(( node, "R" ))) for node in range(N) ]

dt = 0.02

visualize_reaction_diffusion(model, network, dt, plot_node_indices)

