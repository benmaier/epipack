import numpy as np
import matplotlib.pyplot as pl

from collections import Counter
from scipy.interpolate import interp1d

from epipack.distributions import (
        ExpChain,
        fit_chain_by_cdf,
        fit_chain_by_median_and_iqr,
    )

import epipack as epk

mean_C_duration = 11
n = 3

singlechain = ExpChain([mean_C_duration/n for i in range(n)])


def incidence(t,y=None):
    return 100*np.sin(t/10)**10

def get_model(chain,incidence):

    C_compartments = [ 'C_' + str(i) for i in range(len(chain.tau)) ]

    transition_processes = [
                       (None, incidence, 'C_0'),
                    ]
    last_itau = len(chain.tau) - 1

    for itau in range(last_itau):
        transition_processes.append(
                ('C_'+str(itau), 1/chain.tau[itau], 'C_'+str(itau+1))
            )

    transition_processes.append(
            ('C_'+str(last_itau), 1/chain.tau[last_itau], None)
        )

    model = (
                epk.EpiModel(C_compartments)
                   .add_transition_processes(transition_processes)
                   .set_initial_conditions({})
            )

    return model



def get_timepoints_local_maxima(t,y):
    dy = np.diff(y)
    s_dy = np.sign(dy)
    d_s_dy = np.diff(s_dy)
    print
    ndx = np.where(d_s_dy == -2)[0] + 1
    return t[ndx]



def analyze_model(model,last_tau,ax=None):
    dt = 0.1
    tt = np.linspace(0,100,int(100/dt)+1)

    model.set_initial_conditions({})
    res = model.integrate(tt)

    new_analysis = ax is None
    if new_analysis:
        fig, ax = pl.subplots(1,1)
        local_maxima_incidence = get_timepoints_local_maxima(tt,incidence(tt,None))
        ax.plot(tt, incidence(tt,None))

    C_prevalence = None
    for C in model.compartments:
        if C_prevalence is None:
            C_prevalence = res[C]
        else:
            C_prevalence += res[C]

    #ax.plot(tt, C_prevalence)
    last_itau = len(model.compartments)-1
    reported_incidence =  res['C_'+str(last_itau)]/last_tau
    ax.plot(tt, reported_incidence)
    ax.set_xlabel('days')
    ax.set_ylabel('incidence')
    #ax.set_yscale('log')
    local_maxima_reported = get_timepoints_local_maxima(tt,reported_incidence)
    print("difference in peaks=",local_maxima_reported - local_maxima_incidence)

    return ax, reported_incidence


model = get_model(singlechain,incidence)
ax, reported_incidence = analyze_model(model,singlechain.tau[-1])

from scipy.signal import deconvolve
from scipy.stats import gamma
scale = mean_C_duration/n
erlang = gamma(a=n,scale=scale)
tmax = 25
dt = 0.1
t = np.linspace(0,tmax,int(tmax/dt)+1)
pdf = (erlang.cdf(t+dt) - erlang.cdf(t))/dt
deconv_inc, _ = deconvolve(reported_incidence, pdf)
#print(deconv_inc)
#ax.plot(0.1*np.arange(len(deconv_inc)), deconv_inc)

fig = ax.get_figure()
fig.tight_layout()
#fig.savefig('frequency_00.pdf')



pl.show()







