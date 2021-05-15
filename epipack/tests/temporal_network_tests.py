import unittest

import numpy as np
from scipy.optimize import root
from scipy.integrate import cumtrapz
from scipy.stats import entropy, poisson

from epipack import StochasticEpiModel
from epipack.temporal_networks import TemporalNetwork, TemporalNetworkSimulation

class FakeTacomaNetwork():

    t = [0,0.5,0.6]
    edges = [ [ (0,1) ], [ (0,1), (0,2) ], [] ]
    N = 3
    tmax = 1.0

class TemporalNetworkTest(unittest.TestCase):

    def test_temporal_network(self):

        expected = [
            (0, 0.5, [(0, 1, 1.0)]),
            (0.5, 0.6, [(0, 1, 1.0), (0, 2, 1.0)]),
            (0.6, 1.0, []),
            (1.0, 1.5, [(0, 1, 1.0)]),
            (1.5, 1.6, [(0, 1, 1.0), (0, 2, 1.0)]),
            (1.6, 2.0, []),
            (2.0, 2.5, [(0, 1, 1.0)]),
            (2.5, 2.6, [(0, 1, 1.0), (0, 2, 1.0)]),
            (2.6, 3.0, []),
        ]

        edges = [ [ (0,1) ], [ (0,1), (0,2) ], [] ]
        temporal_network = TemporalNetwork(3,edges,[0,0.5,0.6],1.0)
        for (edge_list, t, next_t), (_t, _next_t, _edge_list) in zip(temporal_network, expected):
            if t >= 3.0:
                break
            assert(t == _t)
            assert(next_t == _next_t)
            assert(set(edge_list) == set(_edge_list))


    def test_temporal_gillespie(self,plot=False):

        infection_rate = 1.0
        recovery_rate = 0.2
        model = StochasticEpiModel(["S","I","R"],3)\
                    .set_link_transmission_processes([
                            ("I", "S", infection_rate, "I", "I"),
                        ])\
                    .set_node_transition_processes([
                            ("I", recovery_rate, "R"),
                        ])\
                    .set_node_statuses([1,0,0])

        edges = [ [ (0,1) ], [ (0,1), (0,2) ], [] ]
        temporal_network = TemporalNetwork(3,edges,[0,0.5,1.2],1.5)
        sim = TemporalNetworkSimulation(temporal_network, model)
        N_meas = 10000
        taus = []
        for meas in range(N_meas):
            sim.reset()
            t, res = sim.simulate(1000)
            if t[-1] == 0:
                continue
            else:
                taus.append(t[1])


        def rate(t):
            t = t % 1.5
            if t < 0.5:
                return infection_rate + recovery_rate
            elif t < 1.2:
                return 2*infection_rate + recovery_rate
            elif t < 1.5:
                return recovery_rate


        measured, bins = np.histogram(taus,bins=100,density=True)
        rates = np.array([rate(_t) for _t in bins])
        I2 = cumtrapz(rates,bins,initial=0.0)
        theory = [ np.exp(-I2[i-1])-np.exp(-I2[i]) for i in range(1,len(bins)) if measured[i-1] > 0]
        experi = [ measured[i-1] for i in range(1,len(bins)) if measured[i-1] > 0]
        # make sure the kullback-leibler divergence is below some threshold
        if plot: # pragma: no cover
            import matplotlib.pyplot as pl
            pl.figure()
            pl.hist(taus,bins=100,density=True)
            tt = np.linspace(0,max(taus),10000)
            rates = np.array([rate(_t) for _t in tt])
            I2 = cumtrapz(rates,tt,initial=0.0)
            pl.plot(tt, rates*np.exp(-I2))
            pl.yscale('log')
            pl.figure()
            pl.hist(taus,bins=100,density=True)
            pl.plot(tt, rates*np.exp(-I2))
            pl.show()
        assert(entropy(theory, experi) < 0.02)

    def test_degree(self):

        edges = [ [ (0,1) ], [ (0,1), (0,2) ], [] ]
        temporal_network = TemporalNetwork(3,edges,[0,0.5,1.5],3.0)
        k = temporal_network.mean_out_degree()
        expected = 0.5*np.mean([1,1,0])+1.0*np.mean([2,1,1])
        expected /= 3.0
        assert( np.isclose(k, expected))
        temporal_network = TemporalNetwork(3,edges,[0,0.5,1.5],3.0,directed=True,weighted=False,loop_network=False)
        k = temporal_network.mean_out_degree()
        expected = 0.5*np.mean([1,0,0])+1.0*np.mean([2,0,0])
        expected /= 3.0
        assert( np.isclose(k, expected))
        edges = [ [ (0,1,1.0) ], [ (0,1,0.5), (0,2,2.0) ], [] ]
        temporal_network = TemporalNetwork(3,edges,[0,0.5,1.5],3.0,directed=False,weighted=True)
        k = temporal_network.mean_out_degree()
        expected = 0.5*np.mean([1,1,0.])+1.0*np.mean([2.5,0.5,2.0])
        expected /= 3.0
        assert( np.isclose(k, expected))
        temporal_network = TemporalNetwork(3,edges,[0,0.5,1.5],3.0,directed=True,weighted=True)
        k = temporal_network.mean_out_degree()
        expected = 0.5*np.mean([1,0,0.])+1.0*np.mean([2.5,0,0])
        expected /= 3.0
        assert( np.isclose(k, expected))

    def test_tacoma_network(self):

        expected = [
            (0, 0.5, [(0, 1, 1.0)]),
            (0.5, 0.6, [(0, 1, 1.0), (0, 2, 1.0)]),
            (0.6, 1.0, []),
            (1.0, 1.5, [(0, 1, 1.0)]),
            (1.5, 1.6, [(0, 1, 1.0), (0, 2, 1.0)]),
            (1.6, 2.0, []),
            (2.0, 2.5, [(0, 1, 1.0)]),
            (2.5, 2.6, [(0, 1, 1.0), (0, 2, 1.0)]),
            (2.6, 3.0, []),
        ]

        temporal_network = TemporalNetwork.from_tacoma(FakeTacomaNetwork())
        for (edge_list, t, next_t), (_t, _next_t, _edge_list) in zip(temporal_network, expected):
            if t >= 3.0:
                break
            assert(t == _t)
            assert(next_t == _next_t)
            assert(set(edge_list) == set(_edge_list))



if __name__ == "__main__":

    import sys

    T = TemporalNetworkTest()
    T.test_tacoma_network()
    T.test_temporal_network()
    T.test_degree()
    T.test_temporal_gillespie(plot=True)
