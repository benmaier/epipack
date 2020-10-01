import epipack as epk

N = 1000
S, I, R = list("SIR")
model = epk.EpiModel([S,I,R],
                     initial_population_size=N,
                     correct_for_dynamical_population_size = True
                     )

alpha = 1/2
beta = 1/4
gamma = 1
I0 = 100

model.set_processes([
        (S, I, alpha, I, I),
        (I, beta, R),
        (None, gamma, S),
        (None, 1e-3, I),
    ])
model.set_initial_conditions({S:N-I0, I: I0})

t, result_sim = model.simulate(4000)

from bfmplot import pl
import bfmplot as bp

for iC, (C, res) in enumerate(result_sim.items()):
    pl.plot(t, res,label=C,lw=1.5,c=bp.colors[iC])

model.set_processes([
        (S, I, alpha, I, I),
        (I, beta, R),
        (None, gamma, S),
    ])

result = model.integrate(t)

for iC, (C, res) in enumerate(result.items()):
    pl.plot(t, res, c=bp.colors[iC], ls='-.')

bp.strip_axis(pl.gca())
pl.xlim([0,150])
pl.ylim([0,1000])

pl.xlabel('time [days]')
pl.ylabel('incidence')
pl.legend()

pl.gcf().tight_layout()
pl.gcf().savefig('SIR_birth_model_reimports_zoom.png',dpi=300)


pl.xlim([0,max(t)])
pl.ylim([0,max(result_sim['S'])])

pl.gcf().savefig('SIR_birth_model_reimports_all.png',dpi=300)

pl.yscale('log')
pl.ylim([min(result['I']),max(result_sim['S'])])
pl.gcf().savefig('SIR_birth_model_reimports_all_log.png',dpi=300)
pl.show()
