import unittest

import numpy as np
import sympy
from sympy import symbols, Matrix, Eq, sqrt, FiniteSet, Derivative

from scipy.optimize import root
from scipy.stats import poisson, entropy

from epipack import (
            SymbolicEpiModel,
            SymbolicODEModel,
            SymbolicSISModel,
            SymbolicSIRModel,
            SymbolicSIRSModel,
            SymbolicSIModel,
            get_temporal_interpolation,
            MatrixSISModel,
            MatrixEpiModel,
            SymbolicMatrixEpiModel,
        )

class SymbolicEpiTest(unittest.TestCase):

    def test_compartments(self):
        comps = sympy.symbols("S E I R")
        epi = SymbolicEpiModel(comps)
        assert(all([ i == epi.get_compartment_id(C) for i, C in enumerate(comps) ]))

    def test_linear_rates(self):

        S, E, I, R = sympy.symbols("S E I R")
        epi = SymbolicEpiModel([S,E,I,R])
        epi.add_transition_processes([
                (E, 1, I),
                (I, 1, R),
                ])
        linear_rates = [ E, I ] 
        linear_events = [ Matrix([[0,-1,+1,0]]), Matrix([[0,0,-1,+1]]) ]
        for r0, r1 in zip(linear_rates, epi.linear_rate_functions):
            assert(r0 == r1)            
        for e0, e1 in zip(linear_events, epi.linear_event_updates):
            assert(all([_e0==_e1 for _e0, _e1 in zip(e0, e1)]))


    def test_adding_linear_rates(self):

        S, E, I, R = sympy.symbols("S E I R")
        epi = SymbolicEpiModel([S,E,I,R])

        epi.set_processes([
                (E, 1, I),
                ])

        epi.add_transition_processes([
                (I, 1, R),
                ])
        linear_rates = [ E, I ] 
        linear_events = [ Matrix([[0,-1,+1,0]]), Matrix([[0,0,-1,+1]]) ]
        for r0, r1 in zip(linear_rates, epi.linear_rate_functions):
            assert(r0 == r1)            
        for e0, e1 in zip(linear_events, epi.linear_event_updates):
            assert(all([_e0==_e1 for _e0, _e1 in zip(e0, e1)]))

    def test_quadratic_processes(self):

        S, E, I, R = sympy.symbols("S E I R")
        epi = SymbolicEpiModel([S,E,I,R])
        quadratic_rates = [ I*S ] 
        quadratic_events = [ Matrix([[-1,+1,0,0]]) ]
        epi.add_transmission_processes([
                (S, I, 1, I, E),
            ])
        for r0, r1 in zip(quadratic_rates, epi.quadratic_rate_functions):
            assert(r0 == r1)            
        for e0, e1 in zip(quadratic_events, epi.quadratic_event_updates):
            assert(all([_e0==_e1 for _e0, _e1 in zip(e0, e1)]))

    def test_adding_quadratic_processes(self):

        S, E, I, A, R, rho = sympy.symbols("S E I A R rho")
        epi = SymbolicEpiModel([S,E,I,A,R])

        epi.set_processes([
                (S, I, rho, I, E),
                ])
        epi.add_transmission_processes([
                (S, A, rho, A, E),
                ])
        quadratic_rates = [ I*S*rho, S*A*rho ] 
        quadratic_events = [ Matrix([[-1,+1,0,0,0]]), Matrix([[-1,+1,0,0,0]]) ]
        for r0, r1 in zip(quadratic_rates, epi.quadratic_rate_functions):
            assert(r0 == r1)            
        for e0, e1 in zip(quadratic_events, epi.quadratic_event_updates):
            assert(all([_e0==_e1 for _e0, _e1 in zip(e0, e1)]))

    def test_SIS_with_simulation_restart_and_euler(self):
        S, I = sympy.symbols("S I")
        N = 100
        epi = SymbolicSISModel(infection_rate=2,recovery_rate=1,initial_population_size=N)
        epi.set_initial_conditions({S:0.99*N,I:0.01*N})
        tt = np.linspace(0,100,2)
        result = epi.integrate(tt,[S])
        assert(np.isclose(result[S][-1],N/2))

        tt = np.linspace(0,100,1000)
        result = epi.integrate_and_return_by_index(tt,[S],integrator='euler')
        assert(np.isclose(result[0,-1],N/2))


    def test_basic_analytics(self):
        S, I, R, eta, rho, omega, t = symbols("S I R eta rho omega, t")

        SIRS = SymbolicEpiModel([S,I,R])

        SIRS.set_processes([
            #### transmission process ####
            # S + I (eta)-> I + I
            (S, I, eta, I, I),

            #### transition processes ####
            # I (rho)-> R
            # R (omega)-> S
            (I, rho, R),
            (R, omega, S),

        ])

        odes = SIRS.ODEs()

        expected = [Eq(Derivative(S, t), -I*S*eta + R*omega), Eq(Derivative(I, t), I*(S*eta - rho)), Eq(Derivative(R, t), I*rho - R*omega)]

        assert(all([got==exp for got, exp in zip(odes,expected) ]))

        fixed_points = SIRS.find_fixed_points()
        expected = FiniteSet((S, 0, 0), (rho/eta, R*omega/rho, R))

        assert(all([got==exp for got, exp in zip(fixed_points,expected) ]))

        J = SIRS.jacobian()
        expected = Matrix([[-I*eta, -S*eta, omega], [I*eta, S*eta - rho, 0], [0, rho, -omega]])
        N = SIRS.N_comp
        assert(all([J[i,j]==expected[i,j] for i in range(N) for j in range(N)]))

        eig = SIRS.get_eigenvalues_at_disease_free_state()
        expected = {-omega: 1, eta - rho: 1, 0: 1}

        assert(all([v == expected[k] for k, v in eig.items()]))

    def test_numerical(self):

        S, I, eta, rho = sympy.symbols("S I eta rho")

        epi = SymbolicSISModel(eta, rho)

        epi.set_initial_conditions({S: 1-0.01, I:0.01 })
        epi.set_parameter_values({eta:2})

        self.assertRaises(ValueError, epi.integrate,
                    [0,1]
                    )

        epi = SymbolicSISModel(eta, rho)

        epi.set_initial_conditions({S: 1-0.01, I:0.01 })
        epi.set_parameter_values({eta:2,rho:1})

        tt = np.linspace(0,10,1000)
        result = epi.integrate(tt)

        epi2 = MatrixSISModel(2, 1)

        epi2.set_initial_conditions({"S": 1-0.01, "I":0.01 })

        tt = np.linspace(0,10,1000)
        result2 = epi2.integrate(tt)

        for c0, res0 in result.items():
            assert(np.allclose(res0, result2[str(c0)]))


    def test_time_dependent_rates(self):

        B, t = sympy.symbols("B t")
        epi = SymbolicEpiModel([B])
        epi.add_fission_processes([
                (B, t, B, B),
            ])
        epi.set_initial_conditions({B:1})
        result = epi.integrate([0,3],adopt_final_state=True)
        assert(np.isclose(epi.y0[0], np.exp(3**2/2)))
        epi.set_initial_conditions({B:1})
        result = epi.integrate(np.linspace(0,3,10000),integrator='euler',adopt_final_state=True)
        eul = epi.y0[0]
        real = np.exp(3**2/2)
        assert(np.abs(1-eul/real)<1e-2)

    def test_exceptions(self):

        B, mu, t = sympy.symbols("B mu t")
        epi = SymbolicEpiModel([B])
        epi.add_fission_processes([
                (B, mu, B, B),
            ])
        epi.set_initial_conditions({B:1})

        self.assertRaises(ValueError,epi.integrate,[0,1])

        self.assertRaises(ValueError,SymbolicEpiModel,[t])

        self.assertRaises(ValueError,epi.get_eigenvalues_at_disease_free_state)

    def test_custom_models(self):
        eta = sympy.symbols("eta")
        epi = SymbolicSIModel(eta)
        epi.set_parameter_values({eta:1})
        epi.set_initial_conditions({epi.compartments[0]:0.99, epi.compartments[1]:0.01})
        epi.integrate([0,1000],adopt_final_state=True)
        assert(np.isclose(epi.y0[0],0))


        S, I, R, eta, rho = sympy.symbols("S I R eta rho")
        epi = SymbolicSIRModel(eta,rho)
        epi.set_parameter_values({eta:2,rho:1})
        S0 = 0.99
        epi.set_initial_conditions({S:S0, I:1-S0})
        R0 = 2
        Rinf = lambda x: 1-x-S0*np.exp(-x*R0)
        res = epi.integrate([0,1000])

        theory = root(Rinf,0.5)
        assert(np.isclose(res[R][-1],theory.x[0]))

        epi = SymbolicSISModel(eta, rho, initial_population_size=100)

        epi.set_initial_conditions({S: 99, I:1 })
        epi.set_parameter_values({eta:2,rho:1})

        tt = np.linspace(0,1000,2)
        result = epi.integrate(tt)
        assert(np.isclose(result[S][-1],50))

        S, I, R, eta, rho, omega = sympy.symbols("S I R eta rho omega")
        epi = SymbolicSIRSModel(eta, rho, omega)
        _eta = 2
        _rho = 1
        _omega = 1

        epi.set_initial_conditions({S: 0.99, I:0.01 })
        epi.set_parameter_values({eta:_eta, rho:_rho, omega:_omega})

        tt = np.linspace(0,1000,2)
        result = epi.integrate(tt)
        assert(np.isclose(result[R][-1],(1-_rho/_eta)/(1+_omega/_rho)))

    def test_ODEs_jupyter(self):
        
        eta = sympy.symbols("eta")
        epi = SymbolicSIModel(eta)
        epi.ODEs_jupyter()

    def test_temporal_interpolation(self):

        self.assertRaises(ValueError, get_temporal_interpolation, [0,1],[0,1],interpolation_degree=0)

        t = sympy.symbols("t")
        p = sympy.Piecewise((1, (0 <= t) & (t < 1)), (2,(1<=t) & (t<2)))
        p2 = get_temporal_interpolation([0,1,2],[1,2],0)
        assert(p.subs(t,0.5)==p2.subs(t,0.5))
        assert(p.subs(t,1.5)==p2.subs(t,1.5))
        assert(str(p) == str(p2))

        p = sympy.interpolating_spline(1,t,[0,1,2],[1,2,3])
        p2 = get_temporal_interpolation([0,1,2],[1,2,3],1)
        assert(p.subs(t,0.5)==p2.subs(t,0.5))
        assert(p.subs(t,1.5)==p2.subs(t,1.5))
        assert(str(p) == str(p2))

    def test_changing_population_size(self):

        A, B, C, t = sympy.symbols("A B C t")
        epi = SymbolicEpiModel([A,B,C],10,correct_for_dynamical_population_size=True)
        epi.set_initial_conditions({A:5, B:5})
        epi.set_processes([
                (A, B, 1, C),
            ],allow_nonzero_column_sums=True)

        dydt = epi.dydt()
        assert(dydt[0] == -1*A*B/(A + B + C))

        _, res = epi.simulate(1e9)
        assert(res[C][-1] == 5)

        epi.set_processes([
                (None, 1+sympy.log(1+t), A),
                (A, 1+sympy.log(1+t), B),
                (B, 1+sympy.log(1+t), None),
            ],allow_nonzero_column_sums=True)

        rates, comp_changes = epi.get_numerical_event_and_rate_functions()
        _, res = epi.simulate(200,sampling_dt=0.05)

        vals = np.concatenate([res[A][_>10], res[B][_>10]])
        rv = poisson(vals.mean())
        measured, bins = np.histogram(vals,bins=np.arange(10)-0.5,density=True)
        theory = [ rv.pmf(i) for i in range(0,len(bins)-1) if measured[i] > 0]
        experi = [ measured[i] for i in range(0,len(bins)-1) if measured[i] > 0]
        # make sure the kullback-leibler divergence is below some threshold
        assert(entropy(theory, experi) < 2e-3)
        assert(np.median(res[A]) == 1)



    def test_stochastic_well_mixed(self):

        S, E, I, R = sympy.symbols("S E I R")

        N = 75000
        tmax = 100
        model = SymbolicEpiModel([S,E,I,R],N)
        model.set_processes([
                ( S, I, 2, E, I ),
                ( I, 1, R),
                ( E, 1, I),
            ])
        model.set_initial_conditions({S: N-100, I: 100})

        tt = np.linspace(0,tmax,10)
        result_int = model.integrate(tt)

        t, result_sim = model.simulate(tmax,sampling_dt=1,return_compartments=[S, R])

        for c, res in result_sim.items():
            #print(c, np.abs(1-res[-1]/result_int[c][-1]))
            #print(c, np.abs(1-res[-1]/result_sim[c][-1]))
            assert(np.abs(1-res[-1]/result_int[c][-1]) < 0.05)

    def test_ODE_model(self):

        eta, rho = sympy.symbols("eta rho")

        orig_model = SymbolicSISModel(eta, rho)

        ode_model = SymbolicODEModel(orig_model.ODEs())

        assert(all([eq0 == eq1 for eq0, eq1 in zip(orig_model.ODEs(), ode_model.ODEs())]))

        self.assertRaises(AttributeError, ode_model.simulate)
        self.assertRaises(AttributeError, ode_model.set_linear_events)
        self.assertRaises(AttributeError, ode_model.set_quadratic_events)
        self.assertRaises(AttributeError, ode_model.set_processes)
        self.assertRaises(AttributeError, ode_model.add_linear_events)
        self.assertRaises(AttributeError, ode_model.add_quadratic_events)
        self.assertRaises(AttributeError, ode_model.add_transition_processes)
        self.assertRaises(AttributeError, ode_model.add_fission_processes)
        self.assertRaises(AttributeError, ode_model.add_fusion_processes)
        self.assertRaises(AttributeError, ode_model.add_transmission_processes)




if __name__ == "__main__":

    T = SymbolicEpiTest()
    T.test_ODE_model()
    sys.exit(0)
    T.test_changing_population_size()
    T.test_stochastic_well_mixed()
    T.test_SIS_with_simulation_restart_and_euler()
    T.test_compartments()
    T.test_linear_rates()
    T.test_adding_linear_rates()
    T.test_quadratic_processes()
    T.test_adding_quadratic_processes()
    T.test_basic_analytics()
    T.test_numerical()
    T.test_time_dependent_rates()
    T.test_exceptions()
    T.test_custom_models()
    T.test_ODEs_jupyter()
    T.test_temporal_interpolation()
