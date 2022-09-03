import numpy as np
from epipack.distributions import ExpChain
import matplotlib.pyplot as pl
times = np.array([1.,2])
chain = ExpChain(times)
lambdas = 1/times
t, CDF = chain.get_cdf()
CDF_th = 1 - lambdas[1]/(lambdas[1]-lambdas[0]) * np.exp(-lambdas[0]*t) + lambdas[0]/(lambdas[1]-lambdas[0]) * np.exp(-lambdas[1]*t)

t_perc, percs = chain.get_cdf_at_percentiles([0.25,0.5,0.75])

t, pdf = chain.get_pdf()
pdf_th = lambdas[0]*lambdas[1]/(lambdas[0]-lambdas[1]) * ( np.exp(-lambdas[1]*t) - np.exp(-lambdas[0]*t) )
pl.figure()
pl.plot(t, CDF)
pl.plot(t, CDF_th)
_, CDF2 = chain.get_cdf(t)
pl.plot(t, CDF2)
pl.plot(t_perc, percs,'s')

pl.figure()
pl.plot(t, pdf)
pl.plot(t, pdf_th)
_, pdf2 = chain.get_pdf(t)
pl.plot(t, pdf2)
pl.show()
