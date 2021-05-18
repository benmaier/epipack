"""
Heuristics to fit duration distributions to sums of
exponentially distributed waiting times.
This module has to be considered work in progress.
It works, but can be much improved, e.g. by
putting conditions on the ordering of waiting
times such that the symmetry in parameter space
can be utilized for fitting.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import entropy

_DEFAULT_METHOD = 'RK23'

class ExpChain():
    """
    A class that represents a chain of states
    where the waiting time between consecutive
    states is distributed exponentially with
    predetermined mean. The class can be
    used to fit total waiting time distributions,
    i. e. distributions of sums of exponential
    random variables.

    Parameters
    ==========
    durations : numpy.ndarray of float
        mean waiting times between states in the chain

    Example
    =======

    >>> C = ExpChain([0.2,0.4])
    >>> C.get_mean()
    0.6
    """

    def __init__(self,durations):

        self.n = n = len(durations)
        self.tau = durations
        self.n_st = int(n) + 1
        self.dt = min(durations) / 50.

        data = []
        rows = []
        cols = []

        for i in range(self.n):
            src = i
            trg = i+1
            rows.append(trg)
            cols.append(src)
            data.append(1/self.tau[i])
            rows.append(src)
            cols.append(src)
            data.append(-1/self.tau[i])

        self.W = csr_matrix((data, (rows, cols)), shape=2*[self.n_st])

    def dydt(self,t,y):
        """
        The ODE that performs the transitions between states.
        """
        return self.W.dot(y)

    def get_cdf(self,t=None,percentile_cutoff=0.9999,method=_DEFAULT_METHOD):
        """
        Obtain the cumulative distribution function of the total waiting
        time this chain represents.

        Parameters
        ==========
        t : numpy.ndarray of float, default = None
            Ordered array of time points for which the CDF
            should be returned. If ``None``,
            ``scipy.optimize.solve_ivp`` will choose
            the points itself.
        percentile_cutoff : float, default = 1 - 1e-4
            maximum value of the CDF
        method : str, default = 'RK23'
            This is going to be passed to
            ``scipy.optimize.solve_ivp``.

        Returns
        =======
        t : numpy.ndarray of float
            Ordered array of time points for which the CDF
            was computed.
        cdf : numpy.ndarray of float
            the corresponding values of the cumulative
            distribution function
        """

        y0 = np.zeros(self.n_st)
        y0[0] = 1.0

        stop_condition = lambda t, y: percentile_cutoff - y[-1]
        stop_condition.terminal = True

        result = solve_ivp(
                            fun=self.dydt,
                            y0=y0,
                            t_eval=t,
                            t_span=(0,np.inf),
                            events=stop_condition,
                            method=method,
                          )


        return result.t, result.y[-1,:]

    def get_pdf(self,t=None,percentile_cutoff=0.9999,method=_DEFAULT_METHOD):
        """
        Obtain the probability distribution function of the total waiting
        time this chain represents. Uses :func:`ExpChain.get_cdf`.

        Parameters
        ==========
        t : numpy.ndarray of float, default = None
            Ordered array of time points for which the CDF
            should be returned. If ``None``,
            ``scipy.optimize.solve_ivp`` will choose
            the points itself.
        percentile_cutoff : float, default = 1 - 1e-4
            maximum value of the CDF
        method : str, default = 'RK23'
            This is going to be passed to
            ``scipy.optimize.solve_ivp``.

        Returns
        =======
        tmean : numpy.ndarray of float
            Ordered array of bin midpoints for which the pdf
            was computed.
        pdf : numpy.ndarray of float
            the corresponding values of the pdf.
        df : numpy.ndarray of float
            the corresponding bin sizes
        """
        t, P = self.get_cdf(t=t,percentile_cutoff=percentile_cutoff,method=method)
        dt = np.diff(t)
        dP = np.diff(P)
        tmean = 0.5*(t[1:]+t[:-1])
        return tmean, dP/dt, dt

    def get_cdf_at_percentiles(self,percentiles,method=_DEFAULT_METHOD):
        """
        Obtain the cumulative distribution function of the total waiting
        time this chain represents.

        Parameters
        ==========
        t : numpy.ndarray of float, default = None
            Ordered array of time points for which the CDF
            should be returned. If ``None``,
            ``scipy.optimize.solve_ivp`` will choose
            the points itself.
        percentile_cutoff : float, default = 1 - 1e-4
            maximum value of the CDF
        method : str, default = 'RK23'
            This is going to be passed to
            ``scipy.optimize.solve_ivp``.

        Returns
        =======
        t : numpy.ndarray of float
            Ordered array of time points for which the CDF
            was computed.
        cdf : numpy.ndarray of float
            the corresponding values of the cumulative
            distribution function
        """

        y0 = np.zeros(self.n_st)
        y0[0] = 1.0
        time_values = np.zeros_like(percentiles)
        t = 0

        for i, P in enumerate(percentiles):

            stop_condition = lambda t, y: P - y[-1]
            stop_condition.terminal = True

            result = solve_ivp(
                                fun=self.dydt,
                                y0=y0,
                                t_span=(t,np.inf),
                                events=stop_condition,
                                method=method,
                              )

            y0 = result.y[:,-1]
            t = result.t[-1]
            time_values[i] = t

        return time_values, percentiles

    def get_median_and_iqr(self,method=_DEFAULT_METHOD):
        """
        Returns the median and inter-quartile range
        of the waiting time distribution this chain
        represents.

        Parameters
        ==========
        method : str, default = 'RK23'
            This is going to be passed to
            ``scipy.optimize.solve_ivp``.

        Returns
        =======
        median : float
            the median of the distribution
        iqr : numpy.ndarray of float
            array of length 2 containing
            the inter-quartile range.
        """

        percentiles = [0.25,0.5,0.75]

        time_values, _ = self.get_cdf_at_percentiles(percentiles,method=method)

        return time_values[1], time_values[[0,2]]

    def get_mean(self):
        """
        Returns the mean waiting time of this chain.
        """
        return sum(self.tau)

def fit_chain_by_cdf(n,time_values,cdf,lower=1e-10,upper=1e10,percentile_cutoff=1-1e-15,x0=None):
    """
    Fit a chain of exponentially distributed random variables
    to a distribution where the cdf is known for several time points.

    While there exist statistcally sound measures to quantify
    the distance between
    two distributions, I found that the total mean squared distance
    actually finds decent fits consistently, so this function
    is going to be using that until someone convinces me that
    another distance measure yields better results.

    This whole thing should be considered heuristic
    patch work in any case.

    Parameters
    ==========
    n : int
        number of transitions in the chain
    time_values : numpy.ndarray of float
        Ordered array of time points for which the cdf
        is known.
    cdf : numpy.ndarray of float
        Ordered array of corresponding CDF values.
    lower: float, default = 1e-10
        lower bound of waiting times for each transition
    upper: float, default = 1e10
        upper bound of waiting times for each transition
    percentile_cutoff : float default = 1 - 1e-15
        max value of the CDF that should be integrated to
    x0 : numpy.ndarray of float, default = None
        array of length n that contains initial guesses
        of the chain's waiting times. If ``None``,
        ``x0`` is going to contain the value ``mean/n``
        n times, where ``mean`` is the mean of the
        distribution determined by ``cdf``.

    Returns
    =======
    chain : ExpChain
        The chain that was fit to the given CDF.

    Example
    =======

    >>> median, iqr = 13.184775302968362, ( 7.81098765, 20.86713744)
    >>> fit_C = fit_chain_by_cdf(3,[iqr[0], median, iqr[1]],[0.25,0.5,0.75])
    >>> fit_C.get_median_and_iqr()
    13.183969129892406, array([ 7.8109697 , 20.86699702])
    >>> fit_C.tau
    [9.22794388 0.75881288 5.72462722]

    """

    if n != int(n):
        raise ValueError("`n` must be integer, but is " + str(n))

    dQ = np.diff(cdf)

    cdf = np.array(cdf)

    if x0 is None:
        mean = (np.diff(np.concatenate((np.array([0]),time_values))) * (1 - cdf)).sum()
        x0 = [mean/n] * n

    def cost(x, n, cdf, time_values, percentile_cutoff):
        C  = ExpChain(x)
        _, P = C.get_cdf(time_values,percentile_cutoff=percentile_cutoff)
        if len(P) < len(cdf):
            P = np.concatenate((P,np.ones(len(cdf)-len(P))))
        #return ( ((cdf - P)/cdf)**2).sum()
        return ( ((cdf - P)/cdf)**2).sum()
        #return max(np.abs(cdf - P))
        #return ( ((percentiles - P))**2).sum()
        #dP = np.diff(P)
        #return entropy(np.diff(P), dQ)


    result = minimize(cost, x0, (n, cdf, time_values, percentile_cutoff), bounds=[(lower,upper)]*n)

    return ExpChain(result.x)


def fit_chain_by_median_and_iqr(n,median,iqr,lower=1e-10,upper=1e10):
    """
    Fit a chain of exponentially distributed random variables
    to a distribution where only median and iqr are known.

    Parameters
    ==========
    n : int
        number of transitions in the chain
    median : float
        the median of the distribution to fit to
    iqr : 2-tuple of float
        the inter-quartile range of the distribution to
        fit to
    lower: float, default = 1e-10
        lower bound of waiting times for each transition
    upper: float, default = 1e10
        upper bound of waiting times for each transition

    Returns
    =======
    chain : ExpChain
        The chain that was fit to the median and iqr.

    Example
    =======

    >>> times = [0.3,6.,9,0.4]
    >>> C = ExpChain(times)
    >>> fit_C = fit_chain_by_median_and_iqr(3,*C.get_median_and_iqr())
    >>> C.get_median_and_iqr()
    13.184775302968362, array([ 7.81098765, 20.86713744])
    >>> fit_C.get_median_and_iqr()
    13.183969129892406, array([ 7.8109697 , 20.86699702])
    >>> C.tau
    [0.3, 6.0, 9, 0.4]
    >>> fit_C.tau
    [9.22794388 0.75881288 5.72462722]

    """
    percentiles = np.array([0.25,0.5,0.75])
    time_values = np.array([iqr[0],median,iqr[1]])
    mean = median
    x0 = [mean/n]*n
    return fit_chain_by_cdf(n,time_values,percentiles,lower,upper,x0=x0)


if __name__ == "__main__":
    import matplotlib.pyplot as pl
    from scipy.stats import erlang

    n = 10
    mean = 10
    C = ExpChain([mean/n]*n)
    E = erlang(a=n,scale=mean/n)

    t, y = C.get_cdf()

    pl.plot(t, y)
    pl.plot(t, E.cdf(t))

    print(C.get_median_and_iqr())
    print(E.median(), E.interval(0.5))
    pl.show()


    # =========
    times = [0.3,6.,9,0.4]
    n = len(times)
    C = ExpChain(times)
    fit_C = fit_chain_by_median_and_iqr(3,*C.get_median_and_iqr())

    print("C med iqr =",C.get_median_and_iqr())
    print("fit med iqr =",fit_C.get_median_and_iqr())
    print("C durs =",C.tau)
    print("fit durs =",fit_C.tau)

    fig, ax = pl.subplots(1,2,figsize=(8,4))
    for ch in [C, fit_C]:
        t, P = ch.get_cdf()
        dP = np.diff(P)
        tmean = 0.5*(t[1:] + t[:-1])
        dt = np.diff(t)
        ax[1].bar(tmean, dP/dt,width=dt,alpha=0.5)
        ax[0].plot(t,P)
        print(sum(dP))

    pl.show()


    # ==================

    data = np.array([
    [0.9782608695652173, 0.6117433930093777],
    [1.9565217391304341, 0.7045325376527423],
    [2.934782608695652, 0.7829425973287867],
    [3.9673913043478257, 0.789456048119731],
    [4.945652173913045, 0.8064281992990434],
    [5.923913043478262, 0.7698055792365256],
    [6.9565217391304355, 0.7266458274130909],
    [7.934782608695654, 0.6691081746708346],
    [8.967391304347826, 0.6115693378800797],
    [9.945652173913045, 0.5474957374254049],
    [10.923913043478263, 0.4781933788007957],
    [11.956521739130435, 0.4245761106374918],
    [12.880434782608697, 0.3539677465188974],
    [13.96739130434783, 0.29773491522212747],
    [14.945652173913045, 0.24411883110732224],
    [15.923913043478263, 0.19703869470493518],
    [17.01086956521739, 0.15518494837548535],
    [17.934782608695652, 0.1342497868712702],
    [18.913043478260867, 0.09893435635123615],
    [19.945652173913043, 0.07799682675002373],
    [20.92391304347826, 0.06098204982476085],
    [21.956521739130434, 0.04658046793596648],
    [22.934782608695652, 0.030872880553187487],
    [24.07608695652174, 0.028233636449748856],
    [25, 0.016448801742919406],
    [25.978260869565215, 0.01119873070000943],
    [26.902173913043477, 0.011178601875532879],
    [27.989130434782606, 0.007233352278109395],
    [28.804347826086957, 0.007215591550630007],
    [29.945652173913047, 0.004576347447191376],
    [31.03260869565218, 0.003245476934735203],
    [31.902173913043477, 0.0032265321587572338],
    [32.93478260869565, 0.003204035237283298],
    [33.96739130434783, -0.000740030311641604],
    [35, -0.0007625272331155397],
    [36.141304347826086, 0.0005197972908970172],
    [37.22826086956522, 0.0004961163209245001],
    [38.15217391304348, 0.00047598749644794935],
    [39.07608695652174, 0.001763048214454832],
    [39.94565217391305, -0.0008702756464904482],
    [40.869565217391305, 0.0004167850715164345],
    [41.79347826086956, 0.0017038457895235393],
    [42.93478260869565, -0.0009353983139148703],
    [43.96739130434783, -0.0009578952353888059],
    [44.94565217391305, 0.0003279814341194953],
    ])
    t = np.round(data[:,0])
    print(t)
    P = data[:,1]
    P[P<0] = 0
    P /= P.sum()
    CDF = np.cumsum(P)
    fig, ax = pl.subplots(1,2,figsize=(8,4))
    ax[0].plot(t, CDF)
    ax[1].bar(t, P,alpha=0.4)
    ch = fit_chain_by_cdf(3,t,CDF,lower=0.1,upper=50,x0=[1.29,3.37,4.01])
    #ch = fit_chain_by_cdf(8,t,CDF,lower=0.1,upper=50)
    t, dPdt, width = ch.get_pdf()
    ax[1].bar(t, dPdt, width=width,alpha=0.4)
    ax[0].plot(*ch.get_cdf(t))

    pl.show()



    # ===========

    med_iqr = [
                ( 5, ( 2, 13)),
                (11, ( 6, 21)),
                (10, ( 4, 19)),
                ( 6, ( 3, 11)),
                ( 9, ( 4, 18)),
            ]

    for med, iqr in med_iqr:
        fit_C = fit_chain_by_median_and_iqr(3,med,iqr,lower=0.1,upper=100)
        print("demanded med and iqr =", med, iqr)
        print("fitted   med and iqr =", fit_C.get_median_and_iqr())
        print("fitted durations =",fit_C.tau)
        print()
        fig, ax = pl.subplots(1,2,figsize=(8,4))
        ax[0].plot(*fit_C.get_cdf())
        t, dPdt, width = fit_C.get_pdf()
        ax[1].bar(t, dPdt, width=width)

    pl.show()





