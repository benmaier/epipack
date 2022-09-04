import unittest
import sys

import numpy as np
import scipy.sparse as sprs

from epipack.numeric_matrix_epi_models import (
            NetworkMarkovEpiModel,
            MatrixEpiModel,
        )

from epipack.stochastic_epi_models import (
            StochasticEpiModel,
        )

from epipack.integrators import (
            integrate_dopri5,
        )


_S, _I, _R = range(3)

class SIR_Markov_test():

    def __init__(self,N,edge_weight_tuples,infection_rate,recovery_rate=1.,directed=False):

        self.recovery_rate = recovery_rate
        self.infection_rate = infection_rate
        self.N = N
        self.directed = directed
        self.N_comp = 3
        self.y0 = np.zeros((self.N_comp, self.N))

        self.adjacency_matrix = sprs.lil_matrix((N,N),dtype=float)
        for u, v, w in edge_weight_tuples:
            self.adjacency_matrix[v,u] = w
            if not directed:
                self.adjacency_matrix[u,v] = w
        self.adjacency_matrix = self.adjacency_matrix.tocsr()

    def set_node_statuses(self,node_status):

        assert(len(node_status) == self.N)

        for node, status in enumerate(node_status):

            if status not in [_S,_I,_R]:
                raise ValueError("Mapping must be S = "+str(_S)+", I = "+str(_I)+", R = "+str(_R)+", can't work with status = " + str(status))

            self.y0[status,node] = 1.0

        self.y0 = self.y0.ravel()


    def dydt(self,t,_y):

        y = _y.reshape(self.N_comp, self.N)

        S = y[_S,:]
        I = y[_I,:]
        R = y[_R,:]

        dy = np.zeros_like(y)
        dy[_S,:] = -self.infection_rate * S *(self.adjacency_matrix.dot(I.T))
        dy[_R,:] = self.recovery_rate * I
        dy[_I,:] = -dy[_S,:]
        dy[_I,:] -= dy[_R,:]

        return dy.ravel()

    def integrate(self,tt):
        result = integrate_dopri5(self.dydt, tt, self.y0)
        return result.reshape(self.N_comp,self.N,len(tt))




class NetworkMarkovEpiTest(unittest.TestCase):

    def test_compartments(self):

        model = NetworkMarkovEpiModel(list("SIR"),2,[])
        assert(model.get_compartment(1) == ('S', 1))
        assert(model.get_compartment(4) == ('R', 0))

    def test_main(self,
                  N=2,
                  edge_weight_tuples=[
                        (0,1,1.0),
                      ],
                  infection_rate=2.0,
                  recovery_rate=0.5,
                  directed=True,
                  ):

        alpha = infection_rate
        statuses = [_I] + [_S for i in range(1,N) ]
        SIR = SIR_Markov_test(N,edge_weight_tuples,alpha,directed=directed,recovery_rate=recovery_rate)
        SIR.set_node_statuses(statuses)

        model = (NetworkMarkovEpiModel(list("SIR"), N, edge_weight_tuples,directed=directed)
                    .set_link_transmission_processes([
                            ("I", "S", alpha, "I", "I"),
                        ])
                    .set_node_transition_processes([
                            ("I", recovery_rate, "R"),
                        ])
                    .set_node_statuses(statuses)
                 )

        t = np.linspace(0,10,11)
        result = SIR.integrate(t)
        result = result.sum(axis=1)
        result = {
                    C: result[iC,:] for iC, C in enumerate("SIR")
                 }
        result2 = model.integrate(t)
        result2 = model.collapse_result_to_status_counts(result2)

        for C in "SIR":
            assert(np.all(np.isclose(result[C], result2[C])))

    def test_2_nodes_directed(self):
        self.test_main()

    def test_2_nodes_undirected(self):
        self.test_main(directed=False)

    def test_3_nodes_directed(self,directed=True):
        N = 3
        edge_weight_tuples = [
                    (0, 1, 1.0),
                    (1, 2, 2.0),
                    (2, 0, 3.0),
                ]
        self.test_main(
                    N=N,
                    edge_weight_tuples=edge_weight_tuples,
                    directed=directed,
                )

    def test_3_nodes_undirected(self):
        self.test_3_nodes_directed(directed=False)

    def test_complicated_model_directed(self,directed=True):
        N = 4
        edges = [
                    (0, 1, 1.0),
                    (1, 2, 2.0),
                    (2, 0, 3.0),
                    (2, 3, 2.0),
                ]

        alpha = 2.0
        beta = 1.0
        gamma = 0.1
        delta = 0.5
        scale_symptomatic = 2

        S, E, A, I, R = comps = list("SEAIR")
        def icomp(s):
            return comps.index(s)

        node_statuses = [ icomp("S") if i>0 else icomp("A") for i in range(N) ]

        transmission = [
                    (I, S, scale_symptomatic*alpha, I, E),
                    (A, S, alpha, A, E),
                ]
        transition = [
                    (E, gamma, A),
                    (A, delta, I),
                    (I, beta, R),
                ]

        gilles = ( StochasticEpiModel(comps, N, edges, directed=directed)
                      .set_node_transition_processes(transition)
                      .set_link_transmission_processes(transmission)
                      .set_node_statuses(node_statuses)
                  )
        markov_from_gilles = gilles.get_markovian_clone()

        markov = ( NetworkMarkovEpiModel(comps, N, edges, directed=directed)
                      .set_node_transition_processes(transition)
                      .set_link_transmission_processes(transmission)
                      .set_node_statuses(node_statuses)
                  )


        net_comps = [
                     (C, i) for C in comps for i in range(N)
                    ]

        net_transmission = []
        net_transition = []

        for src, trg, w in edges:
            net_transmission.extend([
                    ( (S,trg), (I,src), w*scale_symptomatic*alpha, (I,src), (E,trg) ),
                    ( (S,trg), (A,src), w*alpha, (A,src), (E,trg) ),
                ])
            if not directed:
                net_transmission.extend([
                        ( (S,src), (I,trg), w*scale_symptomatic*alpha, (I,trg), (E,src) ),
                        ( (S,src), (A,trg), w*alpha, (A,trg), (E,src) ),
                    ])

        init = {}
        for node in range(N):
            net_transition.extend([
                    ( (E,node), gamma, (A, node)),
                    ( (A,node), delta, (I, node)),
                    ( (I,node), beta, (R,node)),
                ])

            init[comps[node_statuses[node]], node] = 1.0

        markov2 = ( MatrixEpiModel(net_comps)
                       .add_transition_processes(net_transition)
                       .add_transmission_processes(net_transmission)
                       .set_initial_conditions(init)
                   )

        tt = np.linspace(0,20,21)

        result = markov.integrate(tt)
        result2 = markov2.integrate(tt)
        result3 = markov_from_gilles.integrate(tt)

        for C in result.keys():
            assert(np.all(np.isclose(result[C], result2[C])))
            assert(np.all(np.isclose(result[C], result3[C])))


    def test_complicated_model_undirected(self):
        self.test_complicated_model_directed(directed=False)

    def test_conversion_to_stochastic_directed(self,directed=True):

        N = 4
        edges = [
                    (0, 1, 1.0),
                    (1, 2, 2.0),
                    (2, 0, 3.0),
                    (2, 3, 2.0),
                ]

        alpha = 2.0
        recovery_rate = 1.0
        statuses = [_I] + [_S for i in range(1,N) ]


        model = (NetworkMarkovEpiModel(list("SIR"), N, edges,directed=directed)
                    .set_link_transmission_processes([
                            ("I", "S", alpha, "I", "I"),
                        ])
                    .set_node_transition_processes([
                            ("I", recovery_rate, "R"),
                        ])
                    .set_node_statuses(statuses)
                 )

        gilles = (StochasticEpiModel(list("SIR"), N, edges,directed=directed)
                    .set_link_transmission_processes([
                            ("I", "S", alpha, "I", "I"),
                        ])
                    .set_node_transition_processes([
                            ("I", recovery_rate, "R"),
                        ])
                    .set_node_statuses(statuses)
                 )

        gilles_from_markov = model.get_stochastic_clone()

        gilles_from_markov.simulate(tmax=10)

        assert(np.all(gilles.node_status==gilles_from_markov.node_status))
        assert(np.all(gilles.out_degree==gilles_from_markov.out_degree))
        assert(np.all(gilles.out_strength==gilles_from_markov.out_strength))

        transit_0 = gilles.transitioning_compartments
        transit_1 = gilles_from_markov.transitioning_compartments
        transm_0 = gilles.transmitting_compartments
        transm_1 = gilles_from_markov.transmitting_compartments

        for _T0, _T1 in zip(sorted(transit_0), sorted(transit_1)):
            assert(_T0 == _T1)
        for _T0, _T1 in zip(sorted(transm_0), sorted(transm_1)):
            assert(_T0 == _T1)

    def test_conversion_to_stochastic_undirected(self):
        self.test_conversion_to_stochastic_directed(directed=False)


if __name__ == "__main__":

    T = NetworkMarkovEpiTest()
    T.test_compartments()
    T.test_2_nodes_directed()
    T.test_2_nodes_undirected()
    T.test_3_nodes_directed()
    T.test_3_nodes_undirected()
    T.test_complicated_model_directed()
    T.test_complicated_model_undirected()
    T.test_conversion_to_stochastic_directed()
    T.test_conversion_to_stochastic_undirected()
