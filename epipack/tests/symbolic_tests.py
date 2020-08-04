import unittest

import numpy as np
import sympy
from sympy import symbols, Matrix, Eq, sqrt, FiniteSet, Derivative

from epipack.symbolic_epi_models import (
            SymbolicEpiModel,
            SymbolicSISModel,
        )

class SymbolicEpiTest(unittest.TestCase):

    def test_compartments(self):
        comps = sympy.symbols("S E I R ")
        epi = SymbolicEpiModel(comps)
        assert(all([ i == epi.get_compartment_id(C) for i, C in enumerate(comps) ]))

    def test_linear_rates(self):

        S, E, I, R = sympy.symbols("S E I R ")
        epi = SymbolicEpiModel([S,E,I,R])
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

        S, E, I, R = sympy.symbols("S E I R ")
        epi = SymbolicEpiModel([S,E,I,R])

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

        S, E, I, R = sympy.symbols("S E I R ")
        epi = SymbolicEpiModel([S,E,I,R])
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
        epi = SymbolicEpiModel([S,E,I,A,R])

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

    def test_functional_rates(self):

        u, v, k, f, t = symbols("u v k f t")

        GS = SymbolicEpiModel([u,v])

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


if __name__ == "__main__":

    T = SymbolicEpiTest()
    T.test_compartments()
    T.test_linear_rates()
    T.test_adding_linear_rates()
    T.test_quadratic_processes()
    T.test_adding_quadratic_processes()
    T.test_basic_analytics()
    T.test_functional_rates()
