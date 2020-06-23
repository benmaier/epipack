import numpy as np
import matplotlib.pyplot as pl
from metapop import SISModel, FluxData, MetaPopulationModel

flux_data = [ 
            ( 'a', 'b', 1 ),
            ( 'b', 'a', 1 ),
       ]
Flux = FluxData(flux_data)

epi_models = {
        'a': SISModel(2,1,population_size=100)\
                .set_initial_conditions({'S': 99, 'I': 1})
                ,
        'b': SISModel(2,1,population_size=100)\
                .set_initial_conditions({'S':100, 'I': 0})
        }

MetaPop = MetaPopulationModel(Flux, epi_models)

t = np.linspace(0,20,101)
r = MetaPop.get_result_by_location(t)

pl.figure()
pl.plot(t,r['a']['S'],c='k',label='$S_a$')
pl.plot(t,r['a']['I'],c='k',ls='--',label='$I_a$')
pl.plot(t,r['a']['N'],c='k',ls='-.',label='$N_a$')
pl.plot(t,r['b']['S'],c='grey',label='$S_b$')
pl.plot(t,r['b']['I'],c='grey',ls='--',label='$I_b$')
pl.plot(t,r['b']['N'],c='grey',ls='-.',label='$N_b$')
pl.legend(handlelength=2)
pl.show()
