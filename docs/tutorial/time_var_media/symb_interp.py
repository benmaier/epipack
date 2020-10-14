from epipack.symbolic_epi_models import get_temporal_interpolation, SymbolicEpiModel
import sympy
from epipack.plottools import plot
from bfmplot import pl
import numpy as np

data = np.array([
    (0.0,  2.00),
    (0.75, 2.68),
    (1.5,  3.00),
    (2.25, 2.78),
    (3.0,  2.14),
    (3.75, 1.43),
    (4.5,  1.02),
    (5.25, 1.14),
    (6.0,  1.72),
])

times, rates = data[:,0], data[:,1]

f = get_temporal_interpolation(times, rates, interpolation_degree=1)

S, I, R, t, rho = sympy.symbols("S I R t rho")

model = SymbolicEpiModel([S,I,R])
model.set_processes([
        (S, I, f, I, I),
        (I, rho, R),
    ])\
    .set_initial_conditions({S:0.99,I:0.01})\
    .set_parameter_values({rho:1})

t = np.linspace(0,6,1000)

result = model.integrate(t)
ax = plot(t,result)
ax.legend()
ax.get_figure().savefig('interp_SIR_symbolic.png',dpi=300)
pl.show()

