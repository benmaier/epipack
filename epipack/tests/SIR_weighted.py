# -*- coding: utf-8 -*-
"""
This module provides functions related to the simulation and
measurement of epidemics.
"""

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import eye
from scipy.linalg import expm
from scipy.sparse.linalg import eigs
from scipy.optimize import minimize


_EPI_S = 0
_EPI_I = 1
_EPI_R = 2

class SIR_weighted:

    infected = []
    force_of_infection_on_node = []
    node_status = []
    rates = np.array([0.0,0.0])

    def __init__(self,N,weighted_edge_tuples,infection_rate,recovery_rate,number_of_initially_infected):

        self.N = N
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.infected = np.random.choice(N,replace=False,size=(number_of_initially_infected,)).tolist()
        self.recovered = []
        self.node_status = np.array([ _EPI_S for n in range(N) ],dtype=int)
        for n in self.infected:
            self.node_status[n] = _EPI_I 


        self.force_of_infection_on_node = np.array([ 0.0 for n in range(N) ])
        self.graph = [ {} for n in range(N) ]
        for u, v, weight in weighted_edge_tuples:
            self.graph[u][v] = weight
            self.graph[v][u] = weight
            if self.node_status[u] == _EPI_S and self.node_status[v] == _EPI_I:
                self.force_of_infection_on_node[u] += weight
            elif self.node_status[v] == _EPI_S and self.node_status[u] == _EPI_I:
                self.force_of_infection_on_node[v] += weight

    def evaluate_rates(self):
        self.rates[0] = self.force_of_infection_on_node.sum() * self.infection_rate
        self.rates[1] = len(self.infected) * self.recovery_rate
        return self.rates

    def recovery_event(self):
        recovering_node = self.infected.pop( np.random.choice(len(self.infected)) )
        self.node_status[recovering_node] = _EPI_R

        self.force_of_infection_on_node[recovering_node] = 0.0
        
        for neigh, weight in self.graph[recovering_node].items():
            if self.node_status[neigh] == _EPI_S:
                self.force_of_infection_on_node[neigh] -= weight
                if self.force_of_infection_on_node[neigh] < 0.0:
                    self.force_of_infection_on_node[neigh] = 0.0

        self.recovered.append(recovering_node)

    def infection_event(self):
        infecting_node = np.random.choice(self.N, 
                                          p=self.force_of_infection_on_node/self.force_of_infection_on_node.sum()
                                         )
        self.node_status[infecting_node] = _EPI_I
        self.force_of_infection_on_node[infecting_node] = 0.0

        for neigh, weight in self.graph[infecting_node].items():
            if self.node_status[neigh] == _EPI_S:
                self.force_of_infection_on_node[neigh] += weight

        self.infected.append(infecting_node)

    def simulation(self,tmax):

        t = 0.0
        time = []
        I = []
        R = []

        while (t < tmax) and len(self.infected) > 0:

            time.append(t)
            I.append(len(self.infected))
            R.append(len(self.recovered))

            r = self.evaluate_rates()
            Lambda = r.sum()

            tau = np.random.exponential(1.0/Lambda)
            event = np.random.choice(len(r),p=r/Lambda)

            if event == 0:
                self.infection_event()
            else:
                self.recovery_event()

            t += tau

        if len(self.infected) == 0 and (t < tmax):
            time.append(t)
            I.append(0)
            R.append(len(self.recovered))


        return np.array(time), np.array(I, dtype=float), np.array(R, dtype=float)


if __name__ == "__main__":

    N = 100
    k = 10
    omega = 1.6
    recovery_rate = 0.1
    R0 = 10
    t_run_total = 1000

