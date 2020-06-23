import numpy as np
import matplotlib.pyplot as pl
from metapop import SIRModel, FluxData, MetaPopulationModel


F = 2
flux_data = [ 
            ( 'a', 'b', F ),
            ( 'b', 'a', 0 ),
       ]
Flux = FluxData(flux_data)

Sa0 = 50
Ra0 = 50
Sb0 = 50
Rb0 = 10
Na = Sa0+Ra0
Nb = Sb0+Rb0

epi_models = {
        'a': SIRModel(0,0,population_size=Na)\
                .set_initial_conditions({'S': Sa0, 'R': Ra0})
                ,
        'b': SIRModel(0,0,population_size=Nb)\
                .set_initial_conditions({'S': Sb0, 'R': Rb0})
        }

MetaPop = MetaPopulationModel(Flux, epi_models)

t = np.linspace(0,50,21)
r = MetaPop.get_result_by_location(t)

theory = (Sb0 + Sa0/Na*F*t) / (F*t + Nb)

pl.figure()
pl.plot(t,r['b']['S']/r['b']['N'],'s',c='grey',label='$s_b$')
pl.plot(t,theory,'-',label='theory')
pl.legend(handlelength=2)
pl.show()
