import numpy as np
from epipack.distributions import ExpChain, fit_chain_by_median_and_iqr
import matplotlib.pyplot as pl

times = [0.3,6.,9,0.4]
C = ExpChain(times)
fit_C = fit_chain_by_median_and_iqr(3,*C.get_median_and_iqr())

print(C.get_median_and_iqr())
print(fit_C.get_median_and_iqr())

pl.plot(*C.get_cdf())
pl.plot(*fit_C.get_cdf())
pl.show()
