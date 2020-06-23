import matplotlib.pyplot as pl
import numpy as np
from metapop import SISModel 

N = 100
epi = SISModel(R0=2,recovery_rate=1,population_size=N)
epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
tt = np.linspace(0,10,100)
result = epi.get_result_dict(tt,['S','I'])

pl.plot(tt, result['S'],label='S')
pl.plot(tt, result['I'],label='I')
pl.legend()

pl.show()

