import numpy as np
from scipy.interpolate import interp1d
from bfmplot import pl
import bfmplot as bp

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

x2 = np.linspace(times[0],times[-1],1000)
interp_modes = ['zero','linear', 'nearest','quadratic']
pl.figure()
pl.plot(times, rates, 's', label='data')
for kind in interp_modes:
    f = interp1d(times, rates, kind=kind)
    pl.plot(x2,f(x2),label=kind)
pl.legend(frameon=False)
bp.strip_axis(pl.gca())
pl.xlabel('time')
pl.ylabel('value')
pl.gcf().tight_layout()
pl.gcf().savefig('interp1d.png',dpi=300)
pl.show()
