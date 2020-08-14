import unittest

import numpy as np
import sympy
from sympy import symbols, Matrix, Eq, sqrt, FiniteSet, Derivative

from scipy.optimize import root

from epipack import (
            SymbolicMatrixSISModel,
            SymbolicMatrixSIRModel,
            SymbolicMatrixSIRSModel,
            SymbolicMatrixSIModel,
            MatrixSISModel,
            MatrixEpiModel,
            SymbolicMatrixEpiModel,
        )

class SymbolicMatrixEpiTest(unittest.TestCase):

    def test_compartments(self):
        comps = sympy.symbols("S E I R")
        epi = SymbolicMatrixEpiModel(comps)
        assert(all([ i == epi.get_compartment_id(C) for i, C in enumerate(comps) ]))

    def test_linear_rates(self):

        S, E, I, R = sympy.symbols("S E I R")
        epi = SymbolicMatrixEpiModel([S,E,I,R])
        epi.add_transition_processes([
                (E, 1, I),
                (I, 1, R),
                ])

        linear_rates = sympy.zeros(4,4)
        linear_rates[1,1] = -1
        linear_rates[2,1] = +1
        linear_rates[2,2] = -1
        linear_rates[3,2] = +1

        N = epi.N_comp
        assert(all([epi.linear_rates[i,j]==linear_rates[i,j] for i in range(N) for j in range(N)]))

    def test_adding_linear_rates(self):

        S, E, I, R = sympy.symbols("S E I R")
        epi = SymbolicMatrixEpiModel([S,E,I,R])

        epi.set_processes([
                (E, 1, I),
                ])

        epi.add_transition_processes([
                (I, 1, R),
                ])

        linear_rates = sympy.zeros(4,4)
        linear_rates[1,1] = -1
        linear_rates[2,1] = +1
        linear_rates[2,2] = -1
        linear_rates[3,2] = +1

        N = epi.N_comp
        assert(all([epi.linear_rates[i,j]==linear_rates[i,j] for i in range(N) for j in range(N)]))

    def test_quadratic_processes(self):

        S, E, I, R = sympy.symbols("S E I R")
        epi = SymbolicMatrixEpiModel([S,E,I,R])
        Q = [ sympy.zeros(4,4) for C in epi.compartments ]
        Q[0][0,2] = -1
        Q[1][0,2] = +1
        epi.add_transmission_processes([
                (S, I,  1, I, E),
                ])
        N = epi.N_comp
        for iM, M in enumerate(epi.quadratic_rates):
            assert(all([M[i,j]==Q[iM][i,j] for i in range(N) for j in range(N)]))

    def test_adding_quadratic_processes(self):

        S, E, I, A, R, rho = sympy.symbols("S E I A R rho")
        epi = SymbolicMatrixEpiModel([S,E,I,A,R])

        Q = [ sympy.zeros(5,5) for C in epi.compartments ]
        Q[0][0,2] = -rho
        Q[0][0,3] = -rho
        Q[1][0,2] = +rho
        Q[1][0,3] = +rho
        epi.set_processes([
                (S, I, rho, I, E),
                ])
        epi.add_transmission_processes([
                (S, A, rho, A, E),
                ])
        N = epi.N_comp
        for iM, M in enumerate(epi.quadratic_rates):
            assert(all([M[i,j]==Q[iM][i,j] for i in range(N) for j in range(N)]))


    def test_basic_analytics(self):
        S, I, R, eta, rho, omega, t = symbols("S I R eta rho omega, t")

        SIRS = SymbolicMatrixEpiModel([S,I,R])

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

    def test_functional_rates(self):

        u, v, k, f, t = symbols("u v k f t")

        GS = SymbolicMatrixEpiModel([u,v])

        GS.set_processes([
            # third-order coupling
            (u, v, v, v, v),
            # birth and death
            (None, f, u),
            (u, f, None),
            (v, f+k, None),
        ],ignore_rate_position_checks=True)


        odes = GS.ODEs()
        expected = [Eq(Derivative(u, t), -f*u + f - u*v**2), Eq(Derivative(v, t), v*(-f - k + u*v))]
        assert(all([got==exp for got, exp in zip(odes,expected) ]))

        fixed_points = GS.find_fixed_points()
        expected = FiniteSet((1, 0), (-(-f + (f + k)*(f/(2*(f + k)) - sqrt(-f*(4*f**2 + 8*f*k - f + 4*k**2))/(2*(f + k))))/f, f/(2*(f + k)) - sqrt(-f*(4*f**2 + 8*f*k - f + 4*k**2))/(2*(f + k))), (-(-f + (f + k)*(f/(2*(f + k)) + sqrt(-f*(4*f**2 + 8*f*k - f + 4*k**2))/(2*(f + k))))/f, f/(2*(f + k)) + sqrt(-f*(4*f**2 + 8*f*k - f + 4*k**2))/(2*(f + k))))

        evals_a, evals_b = [], []
        for got, exp in zip(fixed_points,expected):
            for a, b in zip(got, exp): 
                eval_a = a.evalf(subs={u:0.1,v:0.2,f:0.3,k:0.4})
                eval_b = b.evalf(subs={u:0.1,v:0.2,f:0.3,k:0.4})
                evals_a.append(eval_a)
                evals_b.append(eval_b)

        assert(set(evals_a) == set(evals_b))

        J = GS.jacobian()
        expected = Matrix([[-f - v**2, -2*u*v], [v**2, -f - k + 2*u*v]])
        N = GS.N_comp
        assert(all([J[i,j]==expected[i,j] for i in range(N) for j in range(N)]))

        eig = GS.get_eigenvalues_at_disease_free_state({"u":1})
        expected = {-f: 1, -f - k: 1}
        assert(all([v == expected[k] for k, v in eig.items()]))

    def test_numerical(self):

        S, I, eta, rho = sympy.symbols("S I eta rho")

        epi = SymbolicMatrixSISModel(eta, rho)

        epi.set_initial_conditions({S: 1-0.01, I:0.01 })
        epi.set_parameter_values({eta:2})

        self.assertRaises(ValueError, epi.integrate,
                    [0,1]
                    )

        epi = SymbolicMatrixSISModel(eta, rho)

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
        epi = SymbolicMatrixEpiModel([B])
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
        epi = SymbolicMatrixEpiModel([B])
        epi.add_fission_processes([
                (B, mu, B, B),
            ])
        epi.set_initial_conditions({B:1})

        self.assertRaises(ValueError,epi.integrate,[0,1])

        self.assertRaises(ValueError,SymbolicMatrixEpiModel,[t])

        self.assertRaises(ValueError,epi.get_eigenvalues_at_disease_free_state)

    def test_custom_models(self):
        eta = sympy.symbols("eta")
        epi = SymbolicMatrixSIModel(eta)
        epi.set_parameter_values({eta:1})
        epi.set_initial_conditions({epi.compartments[0]:0.99, epi.compartments[1]:0.01})
        epi.integrate([0,1000],adopt_final_state=True)
        assert(np.isclose(epi.y0[0],0))


        S, I, R, eta, rho = sympy.symbols("S I R eta rho")
        epi = SymbolicMatrixSIRModel(eta,rho)
        epi.set_parameter_values({eta:2,rho:1})
        S0 = 0.99
        epi.set_initial_conditions({S:S0, I:1-S0})
        R0 = 2
        Rinf = lambda x: 1-x-S0*np.exp(-x*R0)
        res = epi.integrate([0,1000])

        theory = root(Rinf,0.5)
        assert(np.isclose(res[R][-1],theory.x[0]))

        epi = SymbolicMatrixSISModel(eta, rho, initial_population_size=100)

        epi.set_initial_conditions({S: 99, I:1 })
        epi.set_parameter_values({eta:2,rho:1})

        tt = np.linspace(0,1000,2)
        result = epi.integrate(tt)
        assert(np.isclose(result[S][-1],50))

        S, I, R, eta, rho, omega = sympy.symbols("S I R eta rho omega")
        epi = SymbolicMatrixSIRSModel(eta, rho, omega)
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
        epi = SymbolicMatrixSIModel(eta)
        epi.ODEs_jupyter()

    def test_changing_population_size(self):

        A, B, C = sympy.symbols("A B C")
        epi = SymbolicMatrixEpiModel([A,B,C],10,correct_for_dynamical_population_size=True)
        epi.set_processes([
                (A, B, 1, C),
            ])

        dydt = epi.dydt()
        assert(dydt[0] == -1*A*B/(A + B + C))



if __name__ == "__main__":

    T = SymbolicMatrixEpiTest()
    T.test_changing_population_size()
    T.test_compartments()
    T.test_linear_rates()
    T.test_adding_linear_rates()
    T.test_quadratic_processes()
    T.test_adding_quadratic_processes()
    T.test_basic_analytics()
    T.test_functional_rates()
    T.test_numerical()
    T.test_time_dependent_rates()
    T.test_exceptions()
    T.test_custom_models()
    T.test_ODEs_jupyter()
