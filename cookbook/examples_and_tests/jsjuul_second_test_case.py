import numpy as np
import matplotlib.pyplot as pl
from metapop import SIRModel, FluxData, MetaPopulationModel


F = 2
flux_data = [ 
            ( 'a', 'b', F ),
            ( 'b', 'a', 0 ),
       ]
Flux = FluxData(flux_data)

Sa0 = 0
Ra0 = 50
Ia0 = 50
Sb0 = 0
Rb0 = 50
Ib0 = 50
Na = Ia0+Ra0
Nb = Ib0+Rb0
beta = recovery_rate = 0.5

epi_models = {
        'a': SIRModel(0,beta,population_size=Na)\
                .set_initial_conditions({'I': Ia0, 'R': Ra0})
                ,
        'b': SIRModel(0,beta,population_size=Nb)\
                .set_initial_conditions({'I': Ib0, 'R': Rb0})
        }

MetaPop = MetaPopulationModel(Flux, epi_models)

t = np.linspace(0,10,21)
r = MetaPop.get_result_by_location(t)

theoryA = Ia0/Na * np.exp(-beta*t)
theoryB = np.exp(-beta*t) * (Ib0+F*t*Ia0/Na) / (F*t + Nb)

pl.figure()
pl.plot(t,r['a']['I']/r['a']['N'],'s',c='grey',label='$i_a$')
pl.plot(t,theoryA,'-',label='theory')
pl.plot(t,r['b']['I']/r['b']['N'],'o',label='$i_b$')
pl.plot(t,theoryB,'-',label='theory')
pl.legend(handlelength=2)
pl.show()
