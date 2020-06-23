import matplotlib.pyplot as pl
import numpy as np
from metapop import SEAIHDRModel, FluxData, MetaPopulationModel

# Define rates 
N = 100
asymptomatic_infection_rate = .6
symptomatic_infection_rate = asymptomatic_infection_rate
asymptomatic_rate = 1/(5.2-2.86)
fraction_never_symptomatic = 0
rate_escaping_asymptomatic = 1/2.86
escape_rate = 1/7.
fraction_requiring_ICU = 0.05
fraction_ICU_patients_succumbing = .42
rate_of_succumbing = 0.14
rate_leaving_ICU = 0.05

# Two-population set up
epi_models = {
    'a' : SEAIHDRModel(asymptomatic_infection_rate,
                       symptomatic_infection_rate, 
                       asymptomatic_rate, 
                       fraction_never_symptomatic,
                       rate_escaping_asymptomatic,
                       escape_rate,
                       fraction_requiring_ICU,
                       fraction_ICU_patients_succumbing,
                       rate_of_succumbing,
                       rate_leaving_ICU,
                       population_size=N*10)\
                    .set_initial_conditions({'S':0.99*N*10,'E':(1-0.99)*N*10,'A':0,'I':0,'H':0,'D':0,'R':0})
                ,
    'b' : SEAIHDRModel(asymptomatic_infection_rate,
                       symptomatic_infection_rate, 
                       asymptomatic_rate, 
                       fraction_never_symptomatic,
                       rate_escaping_asymptomatic,
                       escape_rate,
                       fraction_requiring_ICU,
                       fraction_ICU_patients_succumbing,
                       rate_of_succumbing,
                       rate_leaving_ICU,
                       population_size=N)\
                    .set_initial_conditions({'S':0.99*N,'E':(1-0.99)*N,'A':0,'I':0,'H':0,'D':0,'R':0})

}

flux_data = [ 
            ( 'a', 'b', 1 ),
            ( 'b', 'a', 0 ),
       ]
Flux = FluxData(flux_data)

MetaPop = MetaPopulationModel(Flux, epi_models)

t = np.linspace(0,100,1001)
r = MetaPop.get_result_by_location(t)

pl.plot(t,r['a']['H'],c='k',label='$H_{Hovedstaden}$')
pl.plot(t,r['a']['D'],c='k',ls='--',label='$D_{Hovedstaden}$')

pl.plot(t,r['b']['H'],c='b',label='$H_{Sjaelland}$')
pl.plot(t,r['b']['D'],c='b',ls='--',label='$D_{Sjaelland}$')

pl.legend()

pl.show()
