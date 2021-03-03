import unittest

import numpy as np
import sympy

from epipack.interactive import GeneralInteractiveWidget, InteractiveIntegrator, LogRange, Range
from epipack import SymbolicEpiModel

def assert_dicts_equal(a,b):
    for k, v in a.items():
        assert(np.isclose(v, b[k]))

class InteractiveTest(unittest.TestCase):

    def test_ranges(self):
        r = Range(min=-1,max=1,step_count=1000,value=0.2)
        expected = dict(min=-1,max=1,step=2/1000.,value=0.2)

        assert_dicts_equal(r, expected)

        r = Range(min=-1,max=1)
        expected = dict(min=-1,max=1,step=2/100.,value=0)

        assert_dicts_equal(r, expected)

        r = LogRange(min=0.1,max=10,step_count=1000,value=0.2)
        expected = dict(min=-1,max=1,step=2/1000.,value=0.2,base=10)

        assert_dicts_equal(r, expected)

        r = LogRange(min=0.1,max=10)
        expected = dict(min=-1,max=1,step=2/100.,value=1,base=10)

        assert_dicts_equal(r, expected)

        A = LogRange(0.1,1,value=0.5)
        B = Range(0.1,1,value=0.5)
        
        for R in [A, B]:
            assert(np.isclose(R + 2,2.5))
            assert(np.isclose(2 + R,2.5))
            assert(np.isclose(R - 2,-1.5))
            assert(np.isclose(2 - R,1.5))
            assert(np.isclose(R * 2,1.0))
            assert(np.isclose(2 * R,1.0))
            assert(np.isclose(R / 2,0.25))
            assert(np.isclose(2 / R,4.0))
            assert(np.isclose(R**2,0.25))
            assert(np.isclose(2**R,np.sqrt(2)))

    def test_interactive_integrator(self):
        S, I, R, R0, tau, omega = sympy.symbols("S I R R_0 tau omega")

        I0 = 0.01
        model = SymbolicEpiModel([S,I,R])\
             .set_processes([
                    (S, I, R0/tau, I, I),
                    (I, 1/tau, R),
                    (R, omega, S),
                ])\
             .set_initial_conditions({S:1-I0, I:I0})

        parameters = {
            R0: LogRange(min=0.1,max=10,step_count=1000),
            tau: Range(min=0.1,max=10,value=8.0),
            omega: 1/14
        }

        t = np.linspace(0.01,200,1000)
        integrator = InteractiveIntegrator(model, parameters, t, figsize=(3,4), return_compartments=[S,I])

        keys = sorted([ str(k) for k in integrator.sliders.keys() ])
        assert(all([a==b for a, b in zip(sorted(['R_0', 'tau']), keys) ]))

        integrator.update_parameters()

        keys = sorted([ str(k) for k in integrator.lines.keys() ])
        assert(all([ a==b for a, b in zip(sorted(['I', 'S']), keys) ]))

        class change():
            new = True

        integrator.update_xscale(change())
        assert(integrator.ax.get_xscale() == 'log')
        assert(integrator.ax.get_yscale() == 'linear')
        integrator.update_yscale(change())
        assert(integrator.ax.get_xscale() == 'log')
        assert(integrator.ax.get_yscale() == 'log')

        integrator = InteractiveIntegrator(model, parameters, t, figsize=(3,4))
        keys = sorted([ str(k) for k in integrator.lines.keys() ])
        assert(all([ a==b for a, b in zip(sorted([str(C) for C in model.compartments]), keys) ]))


    def test_general_interactive_widget(self):


        t = np.linspace(0,100,1000)
        def get_trig(omega_0,T):
            return {
                'A': np.sin(2*np.pi*t/T+omega_0),
                'B': np.cos(2*np.pi*t/T+omega_0),
            }

        parameters = {
            'omega_0': Range(0,7,100),
            'T': LogRange(10,1e3,100),
        }
        integrator = GeneralInteractiveWidget(get_trig, parameters, t)

        keys = sorted([ str(k) for k in integrator.sliders.keys() ])
        assert(all([a==b for a, b in zip(sorted(['omega_0', 'T']), keys) ]))

        integrator.update_parameters()

        keys = sorted([ str(k) for k in integrator.lines.keys() ])
        assert(all([ a==b for a, b in zip(sorted(['A', 'B']), keys) ]))

        class change():
            new = True

        integrator.update_xscale(change())
        assert(integrator.ax.get_xscale() == 'log')
        assert(integrator.ax.get_yscale() == 'linear')
        integrator.update_yscale(change())
        assert(integrator.ax.get_xscale() == 'log')
        assert(integrator.ax.get_yscale() == 'log')

        integrator = GeneralInteractiveWidget(get_trig, parameters, t, figsize=(3,4))
        keys = sorted([ str(k) for k in integrator.lines.keys() ])
        assert(all([ a==b for a, b in zip(sorted([str(C) for C in ['A','B']]), keys) ]))




if __name__ == "__main__":

    T = InteractiveTest()
    T.test_ranges()
    T.test_interactive_integrator()
    T.test_general_interactive_widget()
