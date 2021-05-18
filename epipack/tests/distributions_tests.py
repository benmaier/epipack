import unittest

import numpy as np
from scipy.optimize import root

from epipack.distributions import (
            ExpChain,
            fit_chain_by_cdf,
            fit_chain_by_median_and_iqr,
        )

class DistributionsTest(unittest.TestCase):

    def test_init(self):

        times = [0.3,6.,9,0.4]
        C = ExpChain(times)
        assert(np.isclose(sum(times), C.get_mean()))

    def test_fit(self):

        times = [0.3,6.,9,0.4]
        C = ExpChain(times)
        fit_C = fit_chain_by_median_and_iqr(3,*C.get_median_and_iqr())

        medA, (iqrA0, iqrA1) = C.get_median_and_iqr()
        medF, (iqrF0, iqrF1) = fit_C.get_median_and_iqr()

        assert(np.isclose(medA, medF,rtol=1e-3))
        assert(np.isclose(iqrA0, iqrF0,rtol=1e-3))
        assert(np.isclose(iqrA1, iqrF1,rtol=1e-3))

if __name__=="__main__":
    T = DistributionsTest()
    T.test_init()
    T.test_fit()
