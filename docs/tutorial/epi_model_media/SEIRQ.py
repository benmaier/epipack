import epipack as epk

S, E, I, R, Q = list("SEIRQ")
model = epk.EpiModel(compartments=[S,E,I,R,Q])

print([ model.get_compartment_id(C) for C in [S, E, I, R, Q]])
print([ model.get_compartment(iC) for iC in range(5) ])

infection_rate = 1/2
recovery_rate = 1/4
quarantine_rate = 1/6
symptomatic_rate = 1

model.set_processes([
    # S + I (alpha)-> E + I
    (S, I, infection_rate, E, I),
    
    # E (omega)-> I
    (E, symptomatic_rate, I),

    # I (beta)-> R
    (I, recovery_rate, R),

    # I (kappa)-> Q
    (I, quarantine_rate, Q),
])

I0 = 0.01
model.set_initial_conditions({S: 1-I0, I: I0})

import numpy as np
from bfmplot import pl as plt
import bfmplot as bp

t = np.linspace(0,100,1000)
result = model.integrate(t)

plt.figure()
for compartment, incidence in result.items():
    plt.plot(t, incidence, label=compartment)

plt.xlabel('time [days]')
plt.ylabel('incidence')
plt.legend()

bp.strip_axis(plt.gca())


plt.gcf().tight_layout()
plt.savefig('SEIRQ.png',dpi=300)

N = 1000
I0 = 100
model = epk.EpiModel([S,E,I,R,Q],initial_population_size=N)
model.set_processes([
    (S, I, infection_rate, E, I),
    (E, symptomatic_rate, I),
    (I, recovery_rate, R),
    (I, quarantine_rate, Q),
])
model.set_initial_conditions({S: N-I0, I: I0})

t, result = model.simulate(100)

plt.figure()
for compartment, incidence in result.items():
    plt.plot(t, incidence, label=compartment)

plt.xlabel('time [days]')
plt.ylabel('incidence')
plt.legend()

bp.strip_axis(plt.gca())
plt.gcf().tight_layout()
plt.savefig('SEIRQ_sim.png',dpi=300)


tt = np.linspace(0,max(t),1000)
result_int = model.integrate(tt)

for compartment, incidence in result_int.items():
    plt.plot(tt, incidence)

#plt.gcf().tight_layout()
plt.savefig('SEIRQ_sim_compare_int.png',dpi=300)

plt.show()

