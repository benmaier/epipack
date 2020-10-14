import numpy as np
from scipy.interpolate import interp1d
from bfmplot import pl
import bfmplot as bp

x = np.linspace(0,6,9)
y = 2 + np.sin(x)
x2 = np.linspace(0,6,1000)
for _x, _y in zip(x,y):
    print(_x,_y)
interp_modes = ['zero','linear', 'nearest','quadratic']
pl.figure()
pl.plot(x, y, 's', label='data')
for kind in interp_modes:
    f = interp1d(x, y, kind=kind)
    pl.plot(x2,f(x2),label=kind)
pl.legend()
bp.strip_axis(pl.gca())
pl.xlabel('time')
pl.ylabel('value')
pl.gcf().tight_layout()
pl.show()
