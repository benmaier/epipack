Visualizations
--------------

The human eye is a powerful tool for pattern recognition.
Watching a simulation unfold can therefore be more enlightening
than just looking at final plots. ``epipack`` comes with a
light-weight visualization framework that shows an animation of a
simulation of :class:`epipack.stochastic_epi_models.StochasticEpiModel`
that can be found in :func:`epipack.vis`.

Basics
======

A visualization can be started using :func:`epipack.vis.visualize`. A ``model``
object is passed to this function, alongside a stylized network and a time delta
which represents the amount of simulation time that is supposed to pass between
two consecutive visualization updates. Optionally, a config dictionary can
be passed

The function then opens a window that comprises three basic elements:

1. The visualization. Here, the network is displayed including links and after
   each update, colors are updated according to node status changes. Links
   are switched off if nodes are quarantined. If you want to use this 
   functionality, pass the argument ``quarantine_compartments`` (as a list
   of compartments that are considered quarantined).
2. Plots. Here, the counts of each compartment are drawn as time series.
   If you do not want to use this functionality, turn it off with
   ``config['show_curves'] = False``
   Additionally, you can ignore to plot compartment curves by setting 
   ``plot_ignore_compartments = [ "S", ...]``
3. Legend. A color-coded list of compartment names. Optionally, you can 
   turn it off using
   ``config['show_curves'] = False``.

You can customize many aspects of the visualization by adjusting the
corresponding entry in the config dictionary. The default config dictionary
is

.. code:: python

    _default_config = {
                'plot_sampled_curve': True,
                'draw_links':True,            
                'draw_nodes':True,
                'n_circle_segments':16,
                'plot_height':120,
                'bgcolor':'#253237',
                'curve_stroke_width':4.0,
                'node_stroke_width':1.0,
                'link_color': '#4b5a62',
                'node_stroke_color':'#000000',
                'node_color':'#264653',
                'bound_increase_factor':1.0,
                'update_dt':0.04,
                'show_curves':True,
                'draw_nodes_as_rectangles':False,
                'show_legend': True,
                'legend_font_color':'#fafaef',
                'legend_font_size':10,
                'padding':10,
                'compartment_colors': _colors,
                # _colors is a list of 3-tuples containing
                # rgb values in the range of [0, 255]
            }

Note that you can interact with the visualization up to a certain level:

1. Zooming: Use your mouse to scroll
2. Panning: Use your mouse to drag
3. Pausing: "SPACE" key
4. Increasing the sampling time delta: repeatedly pressing the "UP" key
5. Decreasing the sampling time delta: repeatedly pressing the "DOWN" key
    

Network Layout
==============

We will the `"Social circles: Facebook" data from SNAP`_
in the following. First, download the data. In the console, do

.. code:: bash

    wget --no-check-certificate https://snap.stanford.edu/data/facebook_combined.txt.gz
    gunzip facebook_combined.txt.gz

which downloads and extracts the combined dataset. Now, ``epipack`` expects a stylized network
like it's produced by Netwulf_. Style the network manually like so

.. code:: python

    import numpy as np
    import networkx as nx
    import netwulf as nw

    # load edges from txt file and construct Graph object
    edges = np.loadtxt('facebook_combined.txt')
    G = nx.Graph()
    G.add_edges_from(edges)

    # visualize and save visualization
    network, config = nw.visualize(G)
    nw.save("FB.json",network,config)

Now you can simply load the stylized network every time you need it.

Let's use this network style to simulate an SIR model that has an
additional "X" compartment for quarantine of symptomatic individuals.

.. code:: python

    import netwulf as nw

    from epipack.vis import visualize
    from epipack import StochasticEpiModel

    # load network
    network, config, _ = nw.load('/Users/bfmaier/pythonlib/facebook/FB.json')

    # get the network properties
    N = len(network['nodes'])
    edge_list = [ ( link['source'], link['target'], 1.0 ) for link in network['links'] ]

    # define model
    model = StochasticEpiModel(list("SIRX"),
                               N=N,
                               edge_weight_tuples=edge_list,
                               )
    k0 = model.out_degree.mean()
    R0 = 5
    recovery_rate = 1/8
    quarantine_rate = 1.5 * recovery_rate
    infection_rate = R0 * (recovery_rate) / k0

    # usual infection process
    model.set_link_transmission_processes([
            ("I","S",infection_rate,"I","I")
        ])

    # standard SIR dynamic with additional quarantine of symptomatic infecteds
    model.set_node_transition_processes([
            ("I",recovery_rate,"R"),
            ("I",quarantine_rate,"X"),
        ])

    # set initial conditions with a small number of infected
    model.set_random_initial_conditions({'I':20,'S':N-20})

    # in every step of the simulation/visualization, let a time of `sampling_dt` pass
    sampling_dt = 0.12

    # simulate and visualize, do not plot the "S" count,
    # and remove links from nodes that transition to "X"
    visualize(model,
              network,
              sampling_dt,
              ignore_plot_compartments=['S'],
              quarantine_compartments=['X'],
              )

And this is the result:

.. video:: ../_static/fb.mp4
    :width: 500


        

Grid Layout
===========

Sometimes, the positions of a network are not important. If this is the case,
you can simply use a grid layout for the network. You can load the corresponding
layout like so:

.. code:: python

    from epipack.vis import get_grid_layout

    layout = get_grid_layout(number_of_nodes,windowwidth=400)

which will produce a window of width 400px.
For such a layout, it's recommended to draw nodes as rectangles.
You can do this by calling the ``visualize`` function with 
``config['draw_nodes_as_rectangles'] = True``.

You could use this, for instance, to animate a well-mixed system like so

.. code:: python

    from epipack.vis import visualize, get_grid_layout
    from epipack import StochasticSIRModel

    # get the layout
    N = 100 * 100
    layout = get_grid_layout(N)

    # define model
    R0 = 3
    recovery_rate = 1/8
    model = StochasticSIRModel(N,R0,recovery_rate)
    model.set_random_initial_conditions({'I':20,'S':N-20})

    # start visualization where the "S" count won't be shown
    sampling_dt = 0.5

    visualize(model,network,sampling_dt,
              ignore_plot_compartments=['S'],
              config={'draw_nodes_as_rectangles':True}
              )

Which yields

.. video:: ../_static/grid_SIR.mp4
    :width: 400


Lattice simulation
==================

The grid layout can also be more than just showing a well-mixed simulation.
It is the natural presentation of a lattice. In order to simulate on a 2D lattice,
construct the stochastic model with lattice links that you can get from
:func:`epipack.network_models.get_2D_lattice_links`.

.. code:: python

    from epipack import get_2D_lattice_links

    N_side = 100
    N = 100**2
    links = get_2D_lattice_links(N,periodic=True,diagonal_links=True)

    R0 = 3; recovery_rate = 1/8
    model = StochasticSIRModel(N,R0,recovery_rate,
                               edge_weight_tuples=links)

This will produce a lattice network with periodic boundary conditions where nodes
are connected to their diagonal neighbors, as well.

In a simulation, make sure to not draw links because they won't be visible anyway, i.e.

.. code:: python

    config['draw_links'] = False

The complete visualization code:

.. code:: python

    from epipack.vis import visualize, get_grid_layout
    from epipack import StochasticSIRModel, get_2D_lattice_links

    # define links and network layout
    N_side = 100
    N = N_side**2
    links = get_2D_lattice_links(N_side, periodic=True, diagonal_links=True)
    network = get_grid_layout(N)

    # define model
    R0 = 3; recovery_rate = 1/8
    model = StochasticSIRModel(N,R0,recovery_rate,
                               edge_weight_tuples=links)
    model.set_random_initial_conditions({'I':20,'S':N-20})

    sampling_dt = 1

    visualize(model,network,sampling_dt,
            config={
                 'draw_nodes_as_rectangles':True,
                 'draw_links':False,
               }
          )

with result:

.. video:: ../_static/lattice_SIR.mp4
    :width: 400

.. _Netwulf: https://netwulf.readthedocs.io/en/latest/python_api/post_back.html

.. _`"Social circles: Facebook" data from SNAP`: https://snap.stanford.edu/data/egonets-Facebook.html
