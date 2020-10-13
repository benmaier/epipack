import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import newton

def get_event_rates(t, y):
    return y * (0.05 + 0.03 * np.array([ np.cos(t), np.sin(t), np.cos(t)**2, np.sin(t)**2 ]))


def conservative_method(t0, y0, rand=None):
    integrand = lambda _t : get_event_rates(_t, y0).sum()
    integral = lambda _t : quad(integrand, t0, _t)[0]
    if rand is None:
        rand = np.random.rand()
    _1_minus_r = 1 - rand

    rootfunction = lambda _t: - np.log(_1_minus_r) - integral(_t)
    new_t = newton(rootfunction, t0, fprime=lambda _t: -integrand(_t))
    tau = new_t - t0
    return tau

def new_method(t0, y0, rand=None):
    integrand = lambda _t, _y: [get_event_rates(_t, y0).sum()]
    initial = integrand(t0,None)
    if rand is None:
        rand = np.random.rand()
    _1_minus_r = 1 - rand

    rootfunction = lambda _t, _y: - np.log(_1_minus_r) - _y[0]
    rootfunction.terminal = True
    result = solve_ivp(integrand,[t0,np.inf],y0=[0],method='RK23',events=rootfunction) 
    return result.t_events[0][0] - t0

def cum_hist(data):
    score = 1 - np.arange(1,len(data)+1,dtype=float) / len(data)
    return np.sort(data), score

if __name__ == "__main__":
    rand = 0.834053
    t0 = 1.0
    y0 = np.array([0.1,0.2,0.3,0.4])
    print(conservative_method(t0, y0, rand))
    print(new_method(t0, y0, rand))

    from timeit import timeit

    N_meas = 800    
    print("measuring conservative method")
    t = timeit(lambda: conservative_method(t0, y0),number=N_meas)
    print("needed", t)

    print("measuring new method")
    t = timeit(lambda: new_method(t0, y0),number=N_meas)
    print("needed", t)

    from tqdm import tqdm

    tau_con = []
    tau_new = []
    N_meas = 10000
    for meas in tqdm(range(N_meas)):
        tau_con.append(conservative_method(t0, y0))
        tau_new.append(new_method(t0, y0))

    from bfmplot import pl
    from scipy.integrate import cumtrapz

    pl.hist(tau_con,bins=200,density=True,histtype='step')
    pl.hist(tau_new,bins=200,density=True,histtype='step')
    t = np.linspace(0,120,1000)
    lamb = np.array([ get_event_rates(_t, y0).sum() for _t in t])
    integral = cumtrapz(lamb,x=t,initial=0)
    p = lamb*np.exp(-integral)
    pl.plot(t,p)
    pl.xlabel('tau')
    pl.ylabel('pdf')
    #pl.yscale('log')
    pl.gcf().tight_layout()

    pl.figure()

    pl.plot(*cum_hist(tau_con))
    pl.plot(*cum_hist(tau_new))
    pl.plot(t,np.exp(-integral))
    pl.ylabel('P(t>tau)')
    pl.xlabel('tau')
    #pl.yscale('log')
    pl.gcf().tight_layout()


    tau_con = []
    tau_new = []
    N_meas = 10000
    for meas in tqdm(range(N_meas)):
        r = np.random.rand()
        tau_con.append(conservative_method(t0, y0, r))
        tau_new.append(new_method(t0, y0, r))

    residuals = ( np.array(tau_con) - np.array(tau_new) )/ np.array(tau_new)

    pl.figure()
    pl.hist(residuals,bins=100)

    

    pl.show()


    
