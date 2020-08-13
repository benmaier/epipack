from epipack import NumericMatrixBasedEpiModel, get_2D_lattice_links
import numpy as np

from time import time

start = time()
N = 1000
links = [ (i,i+1,1.0) for i in range(N-1) ]
graph = [[] for node in range(N)]
degree = [ 0.0 for node in range(N)]

for u, v, w in links:
    graph[u].append( (v, w) )
    graph[v].append( (u, w) )
    degree[u] += w
    degree[v] += w

base_compartments = "SIR"

compartments = [ (node, C) for node in range(N) for C in "SIR" ]
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

#for u, v, w in links:
#    for C in base_compartments:
#
#        linear_processes.extend([
#                  ( (u, C), w*mobility_rate, (v, C) ),
#                  ( (v, C), w*mobility_rate, (u, C) ),
#            ])
linear_rates = []

for u in range(N):
    ku = degree[u]
    for C in base_compartments:
        linear_rates.append(
            ( (u, C), (u, C), -ku*mobility_rate ),
        )
        for v, w in graph[u]:
            linear_rates.append(
                ( (v, C), (u, C), w * mobility_rate )
            )


print("set processes")

model.set_processes(quadratic_processes+linear_processes,allow_nonzero_column_sums=True)
#model.set_processes(quadratic_processes,allow_nonzero_column_sums=True)
print("set rates")
model.add_linear_rates(linear_rates,allow_nonzero_column_sums=False)

initial_conditions = { ( node, "S" ): 1.0 for node in range(N) } 
initial_conditions[(0, "S")] = 0.99
initial_conditions[(0, "I")] = 0.01
model.set_initial_conditions(initial_conditions,allow_nonzero_column_sums=True)


t = np.linspace(0,20,100)

from bfmplot import pl

return_compartments = [( node, "I" ) for node in range(N)]

print("integrate")
result = model.integrate_and_return_by_index(t,
                                             return_compartments=return_compartments,
                                             )

print("plot")
pl.figure()

for iC, C in enumerate(return_compartments):
    pl.plot(t, result[iC,:])

end = time()

print("needed", end-start,"seconds")
pl.show()


