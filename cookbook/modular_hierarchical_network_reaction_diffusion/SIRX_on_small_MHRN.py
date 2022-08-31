import netwulf as nw

from epipack.vis import visualize
from epipack import StochasticEpiModel

# load network
network, config, _ = nw.load('./MHRN.json')



# get the network properties
N = len(network['nodes'])
edge_list = [ ( link['source'], link['target'], 1.0 ) for link in network['links'] ]

# define model
model = StochasticEpiModel(list("SIRXTQ"),
                           N=N,
                           edge_weight_tuples=edge_list,
                           )
k0 = model.out_degree.mean()
R0 = 5
recovery_rate = 1/8
quarantine_rate = 1.5 * recovery_rate
infection_rate = R0 * (recovery_rate) / k0

R0 = 3
recovery_rate = 1/8
quarantine_rate = 1/16
tracing_rate = 1/2
waning_immunity_rate = 1/14
infection_rate = R0 * (recovery_rate) / k0

# usual infection process
model.set_link_transmission_processes([
        ("I","S",infection_rate,"I","I")
    ])

# standard SIR dynamic with additional quarantine of symptomatic infecteds
model.set_node_transition_processes([
            ("I",recovery_rate,"R"),
            ("I",quarantine_rate,"T"),
            ("T",tracing_rate,"X"),
            ("Q",waning_immunity_rate,"S"),
    ])

model.set_conditional_link_transmission_processes({
    ("T", "->", "X") : [
             ("X","I",0.03,"X","T"),
             ("X","S",0.03,"X","Q"),
             ],
    })
#model.set_random_initial_conditions({'I':20,'S':N-20})

# set initial conditions with a small number of infected
model.set_random_initial_conditions({'I':10,'S':N-10})

# in every step of the simulation/visualization, let a time of `sampling_dt` pass
sampling_dt = 0.1

# simulate and visualize, do not plot the "S" count,
# and remove links from nodes that transition to "X"
visualize(model,
          network,
          sampling_dt,
          ignore_plot_compartments=[],
          quarantine_compartments=['X'],
          )

