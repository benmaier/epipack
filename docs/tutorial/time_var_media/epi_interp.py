from epipack import EpiModel
from scipy.interpolate import interp1d
from epipack.plottools import plot
import matplotlib.pyplot as pl
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

f = interp1d(times, rates, kind='linear', bounds_error=False)

def infection_rate(t,y):
    return f(t)

model = EpiModel(list("SIR"))
model.set_processes([
        ('S', 'I', infection_rate, 'I', 'I'),
        ('I', 1.0, 'R'),
    ])\
    .set_initial_conditions({'S':0.99,'I':0.01})\

t = np.linspace(0,6,1000)

result = model.integrate(t)
ax = plot(t,result)
ax.legend(frameon=False)

model = EpiModel(list("SIR"))
model.set_processes([
        ('S', 'I', 2.0, 'I', 'I'),
        ('I', 1.0, 'R'),
    ])\
    .set_initial_conditions({'S':0.99,'I':0.01})\

t = np.linspace(0,6,1000)

result = model.integrate(t,return_compartments='I')
ax = plot(t,result,ax=ax,curve_label_format='constant rate {}')
ax.set_ylim([0,1])
ax.legend(frameon=False)


ax.get_figure().savefig('interp_SIR_numeric.png',dpi=300)
pl.show()

