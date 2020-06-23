import numpy as np
import matplotlib.pyplot as pl
from metapop import SIRModel, TemporalFluxData, TemporalMetaPopulationModel

flux_data = [ 
            ( 0, 'a', 'b', 2 ),
            ( 0, 'b', 'a', 0 ),
            ( 10, 'a', 'b', 0 ),
            ( 10, 'b', 'a', 2 ),
       ]
Flux = TemporalFluxData(flux_data)

epi_models = {
        'a': SIRModel(2,1,population_size=100)\
                .set_initial_conditions({'S': 99, 'I': 1})\
                .set_compartment_mobility({'R': False})\
                ,
        'b': SIRModel(2,1,population_size=100)\
                .set_initial_conditions({'S':100, 'I': 0})\
                .set_compartment_mobility({'R': False})\
        }

MetaPop = TemporalMetaPopulationModel(Flux, epi_models)

t = np.linspace(0,20,101)
r = MetaPop.get_result_by_location(t)

pl.figure()
pl.plot(t,r['a']['S'],c='k',label='$S_a$')
pl.plot(t,r['a']['I'],c='k',ls='--',label='$I_a$')
pl.plot(t,r['a']['R'],c='k',ls=':',label='$R_a$')
pl.plot(t,r['a']['N'],c='k',ls='-.',label='$N_a$')
pl.plot(t,r['b']['S'],c='grey',label='$S_b$')
pl.plot(t,r['b']['I'],c='grey',ls='--',label='$I_b$')
pl.plot(t,r['b']['R'],c='grey',ls=':',label='$R_b$')
pl.plot(t,r['b']['N'],c='grey',ls='-.',label='$N_b$')
pl.legend(handlelength=2)
pl.show()
