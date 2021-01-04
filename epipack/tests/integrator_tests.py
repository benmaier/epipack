import unittest

import numpy as np

from epipack.integrators import integrate_SDE, IntegrationMixin

def rel_err(a,b):
    return np.sqrt(1-a/b)

class IntegratorTest(unittest.TestCase):

    def test_OU(self):

        ym = 3
        f = 2
        D = [0,1.5]
        std = np.sqrt(D[1]**2/2/f)

        t = np.linspace(0,1,100001)

        def dydt(t,y):
            return -f*(y-ym)

        y0 = [ym,ym]
        y = integrate_SDE(dydt, t, y0, D)

        assert(np.all(np.isclose(y[0],ym)))

        _std = np.std(y[1,:])
        _ym = np.mean(y[1,:])

        assert(rel_err(std,_std))
        assert(rel_err(ym,_ym))

    def test_assertions(self):

        A = IntegrationMixin()
        def _():
            return None
        A.get_numerical_dydt = _

        with self.assertRaises(ValueError):
            # should raise f"Unknown integrator"
            A.integrate_and_return_by_index([0],integrator='sdads')

        with self.assertRaises(ValueError):
            # should raise ValueError("'diffusion_constants' undefined but necessary for SDE integration.")
            A.integrate_and_return_by_index([0],integrator='sde')




if __name__ == "__main__":

    T = IntegratorTest()
    T.test_OU()
    T.test_assertions()
