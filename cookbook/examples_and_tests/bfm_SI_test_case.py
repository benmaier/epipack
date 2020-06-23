import numpy as np
import matplotlib.pyplot as pl
from metapop import SIModel, FluxData, MetaPopulationModel
from metapop.integrators import integrate_dopri5


def dsdt(t,s,eta,s0,F,Na0):
    sb = 1/(1+(1-s0)/s0*np.exp(eta*t))
    return [-eta*(1-s[0])*s[0] + F/(Na0+F*t) * (sb-s[0])]

F = 2
flux_data = [ 
            ( 'a', 'b', 0 ),
            ( 'b', 'a', F ),
       ]
Flux = FluxData(flux_data)

Sa0 = 25
Sb0 = 50
Na = 100
Nb = 100
eta = infection_rate = 0.5

epi_models = {
        'a': SIModel(eta,population_size=Na)\
                .set_initial_conditions({'S': Sa0, 'I': Na-Sa0})
                ,
        'b': SIModel(eta,population_size=Nb)\
                .set_initial_conditions({'S': Sb0, 'I': Nb-Sb0})
        }

MetaPop = MetaPopulationModel(Flux, epi_models)

t = np.linspace(0,10,21)
r = MetaPop.get_result_by_location(t)

theory = integrate_dopri5(dsdt,t,[Sa0/Na], eta, Sb0/Nb,F,Na)

pl.figure()
pl.plot(t,r['a']['S']/r['a']['N'],'s',c='grey',label='$s_b$')
pl.plot(t,theory[0,:],'-',label='theory')
pl.legend(handlelength=2)
pl.show()
