from epipack import MatrixEpiModel

N = 4
nodes = list(range(N))
links = [ (u,u+1,1.0) for u in range(N-1) ]

base_compartments = list("SIR")
compartments = [
        (node, comp) for node in nodes for comp in base_compartments
    ]
model = MatrixEpiModel(compartments)

infection_rate = 3
recovery_rate = 1
mobility_rate = 0.05

quadratic_processes = []
linear_processes = []

for node in nodes:
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

model.set_processes(quadratic_processes+linear_processes)

initial_conditions = { ( node, "S" ): 1.0 for node in nodes } 
initial_conditions[(nodes[0], "S")] = 0.8
initial_conditions[(nodes[0], "I")] = 0.2
model.set_initial_conditions(initial_conditions,allow_nonzero_column_sums=True)

# set compartments for which you want to obtain the
# result
plot_compartments = [ (node, "I") for node in nodes ]

# integrate
import numpy as np
t = np.linspace(0,12,1000)    
result = model.integrate(t,return_compartments=plot_compartments)

# plot result
from bfmplot import pl as plt
plt.figure()

for (node, _), concentration in result.items():
    plt.plot(t, concentration, label='node: '+str(node))

plt.xlabel("time")
plt.ylabel("I")
plt.legend()
plt.gcf().tight_layout()
plt.gcf().savefig("chain_I.png",dpi=300)
plt.show()


