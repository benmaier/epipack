import unittest

import numpy as np
from scipy.optimize import root
from scipy.stats import entropy, poisson

from epipack.numeric_epi_models import (
            DynamicBirthRate,
            ConstantBirthRate,
            DynamicLinearRate,
            ConstantLinearRate,
            DynamicQuadraticRate,
            ConstantQuadraticRate,
            EpiModel,
            SISModel,
            SIModel,
            SIRModel,
            SEIRModel,
            SIRSModel,
        )

from epipack.integrators import time_leap_ivp, time_leap_newton

from epipack.stochastic_epi_models import StochasticEpiModel

class EpiTest(unittest.TestCase):

    def test_compartments(self):
        epi = EpiModel(list("SEIR"))
        assert(all([ i == epi.get_compartment_id(C) for i, C in enumerate("SEIR") ]))
        assert(epi.get_compartment_id("E") == 1)
        assert(epi.get_compartment(1) == "E")

    def test_linear_rates(self):

        epi = EpiModel(list("SEIR"))
        epi.add_transition_processes([
                ("E", 1.0, "I"),
                ("I", 1.0, "R"),
            ])
        linear_rates = [ ConstantLinearRate(1.0,1), ConstantLinearRate(1.0,2) ] 
        linear_events = [ np.array([0,-1,+1,0]), np.array([0,0,-1,+1.]) ]
        for r0, r1 in zip(linear_rates, epi.linear_rate_functions):
            assert(r0(0,[0.1,0.2,0.3,0.4,0.5]) == r1(0, [0.1,0.2,0.3,0.4,0.5]))
        for e0, e1 in zip(linear_events, epi.linear_event_updates):
            assert(all([_e0==_e1 for _e0, _e1 in zip(e0, e1)]))

        epi = EpiModel(list("SEIR"))
        _r0 = lambda t, y: 2+np.cos(t)
        _r1 = lambda t, y: 2+np.sin(t)
        epi.add_transition_processes([
                ("E", _r0,  "I"),
                ("I", _r1,  "R"),
            ])
        linear_rates = [ DynamicLinearRate(_r0,1), DynamicLinearRate(_r1,2) ] 
        linear_events = [ np.array([0,-1,+1,0]), np.array([0,0,-1,+1.]) ]

        for r0, r1 in zip(linear_rates, epi.linear_rate_functions):
            assert(r0(0,[0.1,0.2,0.3,0.4,0.5]) == r1(0, [0.1,0.2,0.3,0.4,0.5]))
        for e0, e1 in zip(linear_events, epi.linear_event_updates):
            assert(all([_e0==_e1 for _e0, _e1 in zip(e0, e1)]))


    def test_adding_linear_rates(self):

        epi = EpiModel(list("SEIR"))
        epi.set_processes([
                ("E", 1.0, "I"),
                ])

        epi.add_transition_processes([
                ("I", 1.0, "R"),
                ])

        linear_rates = [ ConstantLinearRate(1.0,1), ConstantLinearRate(1.0,2) ] 
        linear_events = [ np.array([0,-1,+1,0]), np.array([0,0,-1,+1.]) ]
        for r0, r1 in zip(linear_rates, epi.linear_rate_functions):
            assert(r0(0,[0.1,0.2,0.3,0.4,0.5]) == r1(0, [0.1,0.2,0.3,0.4,0.5]))
        for e0, e1 in zip(linear_events, epi.linear_event_updates):
            assert(all([_e0==_e1 for _e0, _e1 in zip(e0, e1)]))

    def test_quadratic_processes(self):

        epi = EpiModel(list("SEIAR"))
        quadratic_rates = [ ConstantQuadraticRate(1.0,2,0)] 
        quadratic_events = [ np.array([-1,+1,0,0,0.])]
        epi.add_transmission_processes([
                ("S", "I",  1.0, "I", "E"),
            ])
        for r0, r1 in zip(quadratic_rates, epi.quadratic_rate_functions):
            assert(r0(0,[0.1,0.2,0.3,0.4,0.5]) == r1(0, [0.1,0.2,0.3,0.4,0.5]))
        for e0, e1 in zip(quadratic_events, epi.quadratic_event_updates):
            assert(all([_e0==_e1 for _e0, _e1 in zip(e0, e1)]))


    def test_adding_quadratic_processes(self):

        epi = EpiModel(list("SEIAR"))
        quadratic_rates = [ ConstantQuadraticRate(1.0,2,0), ConstantQuadraticRate(1.0,3,0) ] 
        quadratic_events = [ np.array([-1,+1,0,0,0.]), np.array([-1,+1,0,0,0.]) ]
        epi.set_processes([
                ("S", "I",  1.0, "I", "E"),
            ])
        epi.add_transmission_processes([
                ("S", "A",  1.0, "A", "E"),
            ])
        for r0, r1 in zip(quadratic_rates, epi.quadratic_rate_functions):
            assert(r0(0,[0.1,0.2,0.3,0.4,0.5]) == r1(0, [0.1,0.2,0.3,0.4,0.5]))
        for e0, e1 in zip(quadratic_events, epi.quadratic_event_updates):
            assert(all([_e0==_e1 for _e0, _e1 in zip(e0, e1)]))


    def test_SIS_with_simulation_restart_and_euler(self):
        N = 100
        epi = SISModel(infection_rate=2,recovery_rate=1,initial_population_size=N)
        epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
        tt = np.linspace(0,100,2)
        result = epi.integrate(tt,['S'])
        assert(np.isclose(result['S'][-1],N/2))

        tt = np.linspace(0,100,1000)
        result = epi.integrate_and_return_by_index(tt,['S'],integrator='euler')
        assert(np.isclose(result[0,-1],N/2))

    def test_repeated_simulation(self):

        N = 100
        epi = SISModel(infection_rate=2,recovery_rate=1,initial_population_size=N)
        epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
        tt = np.linspace(0,100,100)
        old_t = tt[0]

        for it, t in enumerate(tt[1:]):
            result = epi.integrate_and_return_by_index([old_t,t],integrator='euler',adopt_final_state=True)
            old_t = t

        assert(np.isclose(result[0,-1],N/2))

    def test_birth_death(self):

        epi = EpiModel(list("SIR"))

        R0 = 2
        rho = 1
        mu = 0.2
        eta = R0 * rho

        with self.assertWarns(UserWarning):
            epi.set_processes([
                    ("S", "I", eta, "I", "I"),
                    ("I", rho, "R"),  
                    (None, mu, "S"),
                    ("S", mu, None),
                    ("R", mu, None),
                    ("I", mu, None),
                ])
        epi.set_initial_conditions({'S': 0.8, 'I':0.2 })

        t = [0,1000]
        res = epi.integrate(t)
        assert(np.isclose(res['S'][-1],(mu+rho)/eta))
        assert(np.isclose(res['I'][-1],mu/eta*(eta-mu-rho)/(mu+rho)))

    def test_dynamic_birth(self):

        A = "A"
        epi = EpiModel([A])
        epi.set_initial_conditions({A:1})
        with self.assertWarns(UserWarning):
            epi.set_processes([
                    (None, lambda t, y: 2*t, A),
                ])
        res = epi.integrate([0,5])
        assert(np.isclose(res[A][-1],5**2+1))

    def test_correcting_for_declining_pop_size(self):

        A, B = list("AB")
        epi = EpiModel([A, B],10,correct_for_dynamical_population_size=True)
        epi.add_transition_processes([
                #(None, 0.1, A),
            ])
        epi.add_fusion_processes([
                (A, B, 1, B),
            ])
        epi.set_initial_conditions({B:4, A:6})
        tt = np.linspace(0,30)
        result = epi.integrate(tt)
        #from matplotlib import pyplot as pl
        #pl.plot(tt, result[A], label=A)
        #pl.plot(tt, result[B], label=B)
        epi.correct_for_dynamical_population_size = False
        result = epi.integrate(tt)
        #pl.plot(tt, result[A], label=A)
        #pl.plot(tt, result[B], label=B)
        #pl.legend()
        #pl.show()


    def test_fusion_and_adding_rates(self):

        A, B, C = list("ABC")

        epi = EpiModel(list("ABC"))

        # this should not raise a warning that rates do not sum to zero
        # as it will be actively suppressed
        epi.add_fusion_processes([
                (A, B, 1, C),
            ])

        with self.assertWarns(UserWarning):
            # this should raise a warning that rates do not sum to zero
            epi.add_quadratic_events([
                    ((A, B), 1, [(C, -1),(A, +1)]),
                ])
        # now rates should sum to zero
        epi.add_quadratic_events([
                        ((A, B), 1, [(B, +1)]),
                    ])

        with self.assertWarns(UserWarning):
            # this should raise a warning that rates do not sum to zero
            epi.add_linear_events([
                   ((A,), 1, [(B,-1)]) 
                ])

    def test_initial_condition_warnings(self):

        A, B, C = list("ABC")

        epi = EpiModel(list("ABC"))

        with self.assertWarns(UserWarning):
            # this should raise a warning that rates do not sum to zero
            epi.set_initial_conditions({A:0.1,B:0.2})

        with self.assertWarns(UserWarning):
            # this should raise a warning that initial conditions were set twice
            epi.set_initial_conditions([(A,0.1),(A,0.2)])


    def test_custom_models(self):

        S, I, R = list("SIR")

        eta = 1
        epi = SIModel(eta)
        epi.set_initial_conditions({"S":0.99, "I":0.01})
        epi.integrate([0,1000],adopt_final_state=True)
        assert(np.isclose(epi.y0[0],0))


        
        eta = 2
        rho = 1
        epi = SIRModel(eta,rho)
        S0 = 0.99
        epi.set_initial_conditions({S:S0, I:1-S0})
        R0 = eta/rho
        Rinf = lambda x: 1-x-S0*np.exp(-x*R0)
        res = epi.integrate([0,100])

        SIR_theory = root(Rinf,0.5).x[0]
        assert(np.isclose(res[R][-1],SIR_theory))


        omega = 1
        epi = SEIRModel(eta,rho,omega)
        epi.set_initial_conditions({S:S0, I:1-S0})
        res = epi.integrate([0,100])
        assert(np.isclose(res[R][-1],SIR_theory))
        #======================


        epi = SISModel(eta, rho, initial_population_size=100)

        epi.set_initial_conditions({S: 99, I:1 })

        tt = np.linspace(0,1000,2)
        result = epi.integrate(tt)
        assert(np.isclose(result[S][-1],50))

        epi = SIRSModel(eta, rho, omega)

        epi.set_initial_conditions({S: 0.99, I:0.01 })

        tt = np.linspace(0,1000,2)
        result = epi.integrate(tt)
        assert(np.isclose(result[R][-1],(1-rho/eta)/(1+omega/rho)))

    def test_temporal_gillespie(self,plot=False):

        scl = 40
        def R0(t,y=None):
            return 4+np.cos(t*scl)

        S, I = list("SI")
        N = 100
        rec = 1
        model = EpiModel([S,I], N)
        model.set_processes([
                (S, I, R0, I, I),
                (I, rec, S),
            ])
        I0 = 1
        S0 = N - I0
        model.set_initial_conditions({
                S: S0,
                I: I0,
            })

        taus = []
        N_sample = 10000
        for sample in range(N_sample):
            tau, _ = model.get_time_leap_and_proposed_compartment_changes(0)
            taus.append(tau)

        I = lambda t: (4*t + 1/scl*np.sin(t*scl))
        I2 = lambda t: I(t)*S0*I0/N+I0*rec*t
        pdf = lambda t: (R0(t)*S0*I0/N + I0*rec) * np.exp(-I2(t))
        measured, bins = np.histogram(taus,bins=100,density=True)
        theory = [ np.exp(-I2(bins[i-1]))-np.exp(-I2(bins[i])) for i in range(1,len(bins)) if measured[i-1] > 0]
        experi = [ measured[i-1] for i in range(1,len(bins)) if measured[i-1] > 0]
        # make sure the kullback-leibler divergence is below some threshold
        if plot: # pragma: no cover
            import matplotlib.pyplot as pl
            pl.figure()
            pl.hist(taus,bins=100,density=True)
            tt = np.linspace(0,1,100)
            pl.plot(tt, pdf(tt))
            pl.yscale('log')
            pl.figure()
            pl.hist(taus,bins=100,density=True)
            tt = np.linspace(0,1,100)
            pl.plot(tt, pdf(tt))
            pl.show()
        assert(entropy(theory, experi) < 0.01)

    def test_temporal_gillespie_repeated_simulation(self,plot=False):

        scl = 40
        def R0(t,y=None):
            return 4+np.cos(t*scl)

        S, I = list("SI")
        N = 100
        rec = 1
        model = EpiModel([S,I], N)
        model.set_processes([
                (S, I, R0, I, I),
                (I, rec, S),
            ])
        I0 = 1
        S0 = N - I0
        model.set_initial_conditions({
                S: S0,
                I: I0,
            })

        taus = []
        N_sample = 10000
        if plot:
            from tqdm import tqdm
        else:
            tqdm = lambda x: x
        tt = np.linspace(0,1,100)
        for sample in tqdm(range(N_sample)):
            tau = None
            model.set_initial_conditions({
                    S: S0,
                    I: I0,
                })
            for _t in tt[1:]: 
                time, result = model.simulate(_t,adopt_final_state=True)
                #print(time, result['I'])
                if result['I'][-1] != I0:
                    tau = time[1]
                    break
            #print()
            if tau is not None:
                taus.append(tau)

        I = lambda t: (4*t + 1/scl*np.sin(t*scl))
        I2 = lambda t: I(t)*S0*I0/N+I0*rec*t
        pdf = lambda t: (R0(t)*S0*I0/N + I0*rec) * np.exp(-I2(t))
        measured, bins = np.histogram(taus,bins=100,density=True)
        theory = [ np.exp(-I2(bins[i-1]))-np.exp(-I2(bins[i])) for i in range(1,len(bins)) if measured[i-1] > 0]
        experi = [ measured[i-1] for i in range(1,len(bins)) if measured[i-1] > 0]
        # make sure the kullback-leibler divergence is below some threshold
        if plot:
            import matplotlib.pyplot as pl
            pl.figure()
            pl.hist(taus,bins=100,density=True)
            tt = np.linspace(0,1,100)
            pl.plot(tt, pdf(tt))
            pl.yscale('log')
            pl.figure()
            pl.hist(taus,bins=100,density=True)
            tt = np.linspace(0,1,100)
            pl.plot(tt, pdf(tt))
            pl.show()
        assert(entropy(theory, experi) < 0.01)

    def test_stochastic_well_mixed(self):

        S, E, I, R = list("SEIR")

        N = 75000
        tmax = 100
        model = EpiModel([S,E,I,R],N)
        model.set_processes([
                ( S, I, 2, E, I ),
                ( I, 1, R),
                ( E, 1, I),
            ])
        model.set_initial_conditions({S: N-100, I: 100})

        tt = np.linspace(0,tmax,10000)
        result_int = model.integrate(tt)

        t, result_sim = model.simulate(tmax,sampling_dt=1,return_compartments=[S, R])

        model = StochasticEpiModel([S,E,I,R],N)
        model.set_link_transmission_processes([
                ( I, S, 2, I, E ),
            ])
        model.set_node_transition_processes([
                ( I, 1, R),
                ( E, 1, I),
            ])
        model.set_random_initial_conditions({S: N-100, I: 100})

        t, result_sim2 = model.simulate(tmax,sampling_dt=1,return_compartments=[S, R])

        for c, res in result_sim2.items():
            #print(c, np.abs(1-res[-1]/result_int[c][-1]))
            #print(c, np.abs(1-res[-1]/result_sim[c][-1]))
            assert(np.abs(1-res[-1]/result_int[c][-1]) < 0.05)
            assert(np.abs(1-res[-1]/result_sim[c][-1]) < 0.05)


    def test_stochastic_fission(self):

        A, B, C = list("ABC")

        N = 10
        epi = EpiModel([A,B,C],N,correct_for_dynamical_population_size=True)
        epi.add_fusion_processes([
                (A, B, 1.0, C),
            ])
        epi.set_initial_conditions({ A: 5, B: 5})

        t, res = epi.simulate(1e9)

        assert(res[C][-1] == 5)

    def test_birth_stochastics(self):

        A, B, C = list("ABC")

        epi = EpiModel([A,B,C],10,correct_for_dynamical_population_size=True)
        epi.set_initial_conditions({A:5, B:5})

        epi.set_processes([
                (None, 1, A),
                (A, 1, B),
                (B, 1, None),
            ],allow_nonzero_column_sums=True)

        _, res = epi.simulate(200,sampling_dt=0.05)

        vals = np.concatenate([res[A][_>10], res[B][_>10]])
        rv = poisson(vals.mean())
        measured, bins = np.histogram(vals,bins=np.arange(10)-0.5,density=True)
        theory = [ rv.pmf(i) for i in range(0,len(bins)-1) if measured[i] > 0]
        experi = [ measured[i] for i in range(0,len(bins)-1) if measured[i] > 0]
        # make sure the kullback-leibler divergence is below some threshold
        #for a, b in zip(theory, experi):
        #    print(a,b)
        assert(entropy(theory, experi) < 1e-2)
        assert(np.median(res[A]) == 1)

    def test_sampling_callback(self):
        epi = SIModel(infection_rate=5.0,initial_population_size=100)
        epi.set_initial_conditions({"S":90,"I":10})
        self.assertRaises(ValueError,epi.simulate,1,sampling_callback=lambda x: x)

        i = 0
        samples = []
        def sampled():
            samples.append(epi.y0[0])

        t, res = epi.simulate(10,sampling_dt=0.1,sampling_callback=sampled)

        assert(all([a==b for a, b in zip(res['S'], samples)]))


    def test_integral_solvers(self):

        def get_event_rates(t, y):
            return y * (0.05 + 0.03 * np.array([ np.cos(t), np.sin(t), np.cos(t)**2, np.sin(t)**2 ]))

        rand = 0.834053
        t0 = 1.0
        y0 = np.array([0.1,0.2,0.3,0.4])
        t_nwt = time_leap_newton(t0, y0, get_event_rates, rand)
        t_ivp = time_leap_ivp(t0, y0, get_event_rates, rand)
        expected = 30.76 
        numeric = np.array([t_nwt, t_ivp])
        assert(np.all( np.abs(numeric-expected)/numeric < 1e-3) )




if __name__ == "__main__":

    import sys

    T = EpiTest()
    T.test_integral_solvers()
    T.test_temporal_gillespie_repeated_simulation()
    T.test_sampling_callback()
    T.test_birth_stochastics()
    T.test_stochastic_fission()
    T.test_correcting_for_declining_pop_size()
    T.test_dynamic_birth()
    T.test_stochastic_well_mixed()
    T.test_temporal_gillespie()
    T.test_compartments()
    T.test_linear_rates()
    T.test_adding_linear_rates()
    T.test_quadratic_processes()
    T.test_adding_quadratic_processes()
    T.test_SIS_with_simulation_restart_and_euler()
    T.test_repeated_simulation()
    T.test_custom_models()
    T.test_birth_death()
    T.test_fusion_and_adding_rates()
    T.test_initial_condition_warnings()
