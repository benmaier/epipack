import numpy as np
import matplotlib.pyplot as pl
from metapop import SIRModel, FluxData, MetaPopulationModel


F = 2
flux_data = [ 
            ( 'a', 'b', 0 ),
            ( 'b', 'a', F ),
       ]
Flux = FluxData(flux_data)

Ia0 = 100
Ra0 = 0
Ib0 = 50
Rb0 = 0
Na = Ia0+Ra0
Nb = Ib0+Rb0
mu = recovery_rate = 0.1

epi_models = {
        'a': SIRModel(0,mu,population_size=Na)\
                .set_initial_conditions({'I': Ia0, 'R': Ra0})
                ,
        'b': SIRModel(0,mu,population_size=Nb)\
                .set_initial_conditions({'I': Ib0, 'R': Rb0})
        }

MetaPop = MetaPopulationModel(Flux, epi_models)

t = np.linspace(0,10,21)
r = MetaPop.get_result_by_location(t)

theory = Ib0/Nb * F*t*np.exp(-mu*t) + Ia0 * np.exp(-mu*t)

pl.figure()
pl.plot(t,r['a']['I'],'s',c='grey',label='$s_b$')
pl.plot(t,r['a']['N'],'o',label='$s_b$')
pl.plot(t,theory,'-',label='theory')
pl.legend(handlelength=2)
pl.show()
