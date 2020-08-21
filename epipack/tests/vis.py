import unittest

import numpy as np

from epipack.vis import (
            visualize_reaction_diffusion,
            visualize,
            get_grid_layout,
            get_random_layout,
        )

from epipack import StochasticSIRModel, StochasticEpiModel, get_2D_lattice_links
import pyglet
import gc

class VisTest(unittest.TestCase):

    def test_grid_well_mixed(self):
        N = 20 * 20
        network = get_grid_layout(N)

        R0 = 3
        recovery_rate = 1/8
        model = StochasticSIRModel(N,R0,recovery_rate)
        model.set_random_initial_conditions({'I':20,'S':N-20})

        sampling_dt = 2

        visualize(model,network,sampling_dt,
                  ignore_plot_compartments=['S'],
                  config={
                        'draw_nodes_as_rectangles':True,
                        'show_legend':False,
                     }
                  )

    def test_reaction_diffustion(self):
        pass

    def test_lattice(self):
        # define links and network layout
        N_side = 20
        N = N_side**2
        links = get_2D_lattice_links(N_side, periodic=True, diagonal_links=True)
        network = get_grid_layout(N)

        # define model
        R0 = 3; recovery_rate = 1/8
        model = StochasticSIRModel(N,R0,recovery_rate,
                                   edge_weight_tuples=links)
        model.set_random_initial_conditions({'I':20,'S':N-20})

        sampling_dt = 2

        visualize(model,network,sampling_dt,
                    config={
                     'draw_nodes_as_rectangles':True,
                     'draw_links':False,
                     'show_legend':False,
                     'show_curves':False,
                   }
              )

    def test_network(self):
        N = 1000
        k = 4
        links = []
        for i in range(N):
            neighs = np.random.randint(0,N-1,size=(k,),dtype=int)
            neighs[neighs>=i] += 1
            for neigh in neighs:
                links.append((i,int(neigh),1.0))

        network = get_random_layout(N,links,windowwidth=500)

        model = StochasticEpiModel(list("SIRXTQ"),
                                   N=len(network['nodes']),
                                   directed=True,
                                   edge_weight_tuples=links,
                                   )
        k0 = model.out_degree.mean()
        R0 = 10
        recovery_rate = 1/8
        quarantine_rate = 1/16
        tracing_rate = 1/2
        waning_immunity_rate = 1/14
        infection_rate = R0 * (recovery_rate) / k0
        model.set_node_transition_processes([
                ("I",recovery_rate,"R"),
                ("I",quarantine_rate,"T"),
                ("T",tracing_rate,"X"),
                ("Q",waning_immunity_rate,"S"),
                ("X",recovery_rate,"R"),
                ])
        model.set_link_transmission_processes([("I","S",infection_rate,"I","I")])
        model.set_conditional_link_transmission_processes({
            ("T", "->", "X") : [
                     ("X","I",0.5,"X","T"),
                     #("X","S",0.5,"X","Q"),
                     ],
            })
        model.set_random_initial_conditions({'I':20,'S':N-20})

        sampling_dt = 0.08

        visualize(model,network,sampling_dt,ignore_plot_compartments=['S'],quarantine_compartments=['X', 'T', 'Q'])


if __name__ == "__main__":

    T = VisTest()
    T.test_network()
    T.test_lattice()
    T.test_grid_well_mixed()
