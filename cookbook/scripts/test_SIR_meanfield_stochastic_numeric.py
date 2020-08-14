from bfmplot import pl
import numpy as np
from epipack.numeric_epi_models import EpiModel
from epipack import StochasticEpiModel

from time import time

S, E, I, R = list("SEIR")

N = 200000
tmax = 50
model = EpiModel([S,E,I,R],N)
model.set_processes([
        ( S, I, 2, E, I ),
        ( I, 1, R),
        ( E, 1, I),
    ])
model.set_initial_conditions({S: N-100, I: 100})

tt = np.linspace(0,tmax,10000)
result_int = model.integrate(tt)

for c, res in result_int.items():
    pl.plot(tt, res)


start = time()
t, result_sim = model.simulate(tmax,sampling_dt=1)
end = time()

print("numeric model needed", end-start, "s")

for c, res in result_sim.items():
    pl.plot(t, res, '--')

model = StochasticEpiModel([S,E,I,R],N)
model.set_link_transmission_processes([
        ( I, S, 2, I, E ),
    ])
model.set_node_transition_processes([
        ( I, 1, R),
        ( E, 1, I),
    ])
model.set_random_initial_conditions({S: N-100, I: 100})

start = time()
t, result_sim = model.simulate(tmax,sampling_dt=1)
end = time()

for c, res in result_sim.items():
    pl.plot(t, res, ':')
print("stochastic model needed", end-start, "s")

pl.show()



