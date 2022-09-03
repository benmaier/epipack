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
        #print(medA, (iqrA0, iqrA1))
        #print(medF, (iqrF0, iqrF1))

        assert(np.isclose(medA, medF,rtol=1e-3))
        assert(np.isclose(iqrA0, iqrF0,rtol=1e-3))
        assert(np.isclose(iqrA1, iqrF1,rtol=1e-3))

    def test_cdf_std(self):
        tau = np.array([1.,2])
        chain = ExpChain(tau)
        lambdas = 1/tau
        t, CDF = chain.get_cdf()
        CDF_th = 1 - lambdas[1]/(lambdas[1]-lambdas[0]) * np.exp(-lambdas[0]*t) \
                   + lambdas[0]/(lambdas[1]-lambdas[0]) * np.exp(-lambdas[1]*t)
        _, CDF2 = chain.get_cdf(t)

        for num, th, num2 in zip(CDF, CDF_th, CDF2):
            assert(np.isclose(num, th))
            assert(np.isclose(num, num2))

    def test_pdf_std(self):
        tau = np.array([1.,2])
        chain = ExpChain(tau)
        lambdas = 1/tau
        t, pdf = chain.get_pdf()
        pdf_th = lambdas[0]*lambdas[1]/(lambdas[0]-lambdas[1]) * \
                 ( np.exp(-lambdas[1]*t) - np.exp(-lambdas[0]*t) )
        _, pdf2 = chain.get_pdf(t)

        for num, th, num2 in zip(pdf, pdf_th, pdf2):
            assert(np.isclose(num, th))
            assert(np.isclose(num, num2))

    def test_cdf_ivp(self):
        tau = np.array([1.,2])
        chain = ExpChain(tau)
        lambdas = 1/tau
        t, CDF = chain.get_cdf_ivp()
        CDF_th = 1 - lambdas[1]/(lambdas[1]-lambdas[0]) * np.exp(-lambdas[0]*t) \
                   + lambdas[0]/(lambdas[1]-lambdas[0]) * np.exp(-lambdas[1]*t)
        _, CDF2 = chain.get_cdf_ivp(t)

        for num, th, num2 in zip(CDF, CDF_th, CDF2):
            assert(np.isclose(num, th))
            assert(np.isclose(num, num2))

    def test_pdf_ivp(self):
        tau = np.array([1.,2])
        chain = ExpChain(tau)
        lambdas = 1/tau
        t, pdf, _ = chain.get_pdf_ivp()
        pdf_th = lambdas[0]*lambdas[1]/(lambdas[0]-lambdas[1]) * \
                 ( np.exp(-lambdas[1]*t) - np.exp(-lambdas[0]*t) )

        for num, th in zip(pdf, pdf_th):
            assert(np.isclose(num, th, rtol=5e-2))

    def test_get_cdf_at_percentiles(self):

        tau = np.array([1.,2])
        chain = ExpChain(tau)
        PERCENTILES = [0.25,0.5,0.75]
        t_perc, percs = chain.get_cdf_at_percentiles(PERCENTILES)

        for num, th in zip(percs, PERCENTILES):
            assert(np.isclose(num, th))


if __name__=="__main__":
    T = DistributionsTest()
    T.test_cdf_ivp()
    T.test_pdf_ivp()
    T.test_init()
    T.test_fit()
    T.test_cdf_std()
    T.test_pdf_std()
    T.test_get_cdf_at_percentiles()
