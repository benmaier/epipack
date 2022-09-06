import numpy as np
import matplotlib.pyplot as pl
import polars as po

from collections import Counter
from scipy.interpolate import interp1d

df = po.read_csv('distribution_icu_world.csv')

days = df['x'].cast(float).to_numpy()

t = np.arange(0, int(max(days))+1,)

counter = Counter(days)
hist = np.zeros(len(t),dtype=float)
for _t in t:
    hist[_t] = counter[int(_t)] / float(len(days))



fig, axh = pl.subplots(1,2,figsize=(8,4),sharex=True)
axh[0].plot(t+0.5, hist,'-s')
axh[1].plot(t+0.5, hist,'-s')
axh[1].set_yscale('log')


# Flavor 1
interp_t = np.concatenate(( t, [t[-1]+1]))
interp_hist = np.concatenate(( hist, [0.]))
interpolated_pdf = interp1d(interp_t,interp_hist,kind='zero',bounds_error=False,fill_value=0.)

tt = np.linspace(0,61,1001)
axh[0].plot(tt, interpolated_pdf(tt))
axh[1].plot(tt, interpolated_pdf(tt))
# Flavor 2
interp_t = np.concatenate(( [0.], t+0.5, [t[-1]+1]))
interp_hist = np.concatenate(( [0.],hist, [0.]))
new_norm = np.trapz(interp_hist,interp_t)
interpolated_pdf = interp1d(interp_t,interp_hist/new_norm,kind='linear',bounds_error=False,fill_value=0.)

tt = np.linspace(0,61,1001)
axh[0].plot(tt, interpolated_pdf(tt))
axh[1].plot(tt, interpolated_pdf(tt))


l25, med, u75 = np.percentile(days+0.5,[25,50,75])

print(f"{np.mean(days)=}")
mean_ICU_duration = np.mean(days+0.5)

print(l25,med,u75)




from epipack.distributions import (
        ExpChain,
        fit_chain_by_cdf,
        fit_chain_by_median_and_iqr,
    )

import epipack as epk

chain = fit_chain_by_median_and_iqr(n=3,median=med,iqr=(l25,u75),callback=True)
print(chain.get_median_and_iqr())
print(chain.tau)

_, pdf = chain.get_pdf(t)
axh[0].plot(t, pdf)
axh[1].plot(t, pdf)

#=================

#cdf = np.cumsum(hist)
#
#chain2 = fit_chain_by_cdf(3,t[:-1],cdf[:-1],callback=True)
#print(chain.get_median_and_iqr())
#
#_, pdf = chain.get_pdf(t)
#
#ax[0].plot(t, pdf)
#ax[1].plot(t, pdf)

#pl.show()

fig, ax = pl.subplots(1,1)

def incidence(t,y=None):
    return 100*np.sin(t/10)**2

def get_model(chain,incidence):

    ICU_compartments = [ 'ICU_' + str(i) for i in range(len(chain.tau)) ]

    transition_processes = [
                       (None, incidence, 'ICU_0'),
                    ]
    last_itau = len(chain.tau) - 1

    for itau in range(last_itau):
        transition_processes.append(
                ('ICU_'+str(itau), 1/chain.tau[itau], 'ICU_'+str(itau+1))
            )

    transition_processes.append(
            ('ICU_'+str(last_itau), 1/chain.tau[last_itau], None)
        )

    model = (
                epk.EpiModel(ICU_compartments)
                   .add_transition_processes(transition_processes)
                   .set_initial_conditions({})
            )

    return model





def analyze_model(model,last_tau,ax=None):
    tt = np.linspace(0,100,1001)

    model.set_initial_conditions({})
    res = model.integrate(tt)

    new_analysis = ax is None
    if new_analysis:
        fig, ax = pl.subplots(1,1)
        ax.plot(tt, incidence(tt,None))

    ICU_prevalence = None
    for C in model.compartments:
        if ICU_prevalence is None:
            ICU_prevalence = res[C]
        else:
            ICU_prevalence += res[C]

    ax.plot(tt, ICU_prevalence)
    last_itau = len(model.compartments)-1
    ax.plot(tt, res['ICU_'+str(last_itau)]/last_tau)

    return ax


model = get_model(chain,incidence)
ax = analyze_model(model,chain.tau[-1])


singlechain = ExpChain([mean_ICU_duration])
_, pdf = singlechain.get_pdf(t)
axh[0].plot(t, pdf)
axh[1].plot(t, pdf)

model = get_model(singlechain,incidence)
analyze_model(model,singlechain.tau[-1],ax=ax)


#=================================

#fig, ax = pl.subplots(1,1)

tt = np.linspace(0,100,1001)
dt = tt[1]-tt[0]
#tt = np.arange(0,100)
result = np.convolve(incidence(tt), interpolated_pdf(tt)*dt)

#ax.plot(tt, incidence(tt,None))
ax.plot(tt, result[:len(tt)])

print(incidence(tt,None))
print(interpolated_pdf(tt))


f = interp1d(tt, result[:len(tt)], kind='linear',bounds_error=False,fill_value=0)
outflux = lambda t, y: -f(t)

model = epk.EpiModel(['ICU']).add_transition_processes([
            (None, incidence, 'ICU'),
            (None, outflux, 'ICU'),
        ]).set_initial_conditions({
        })

res = model.integrate(tt)

print(res)
ax.plot(tt, res['ICU'])

pl.show()







