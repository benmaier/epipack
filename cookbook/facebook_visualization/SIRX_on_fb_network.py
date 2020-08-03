import netwulf as nw

from epipack.vis import visualize
from epipack import StochasticEpiModel

# load network
network, config, _ = nw.load('/Users/bfmaier/pythonlib/facebook/FB.json')

# get the network properties
N = len(network['nodes'])
edge_list = [ ( link['source'], link['target'], 1.0 ) for link in network['links'] ]

# define model
model = StochasticEpiModel(list("SIRX"),
                           N=N,
                           edge_weight_tuples=edge_list,
                           )
k0 = model.out_degree.mean()
R0 = 5
recovery_rate = 1/8
quarantine_rate = 1.5 * recovery_rate
infection_rate = R0 * (recovery_rate) / k0

# usual infection process
model.set_link_transmission_processes([
        ("I","S",infection_rate,"I","I")
    ])

# standard SIR dynamic with additional quarantine of symptomatic infecteds
model.set_node_transition_processes([
        ("I",recovery_rate,"R"),
        ("I",quarantine_rate,"X"),
    ])

# set initial conditions with a small number of infected
model.set_random_initial_conditions({'I':20,'S':N-20})

# in every step of the simulation/visualization, let a time of `sampling_dt` pass
sampling_dt = 0.12

# simulate and visualize, do not plot the "S" count,
# and remove links from nodes that transition to "X"
visualize(model,
          network,
          sampling_dt,
          ignore_plot_compartments=['S'],
          quarantine_compartments=['X'],
          )

