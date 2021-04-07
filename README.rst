.. image:: https://github.com/benmaier/epipack/raw/master/img/logo_12_lila_medium.png
   :alt: logo

Fast prototyping of epidemiological models based on reaction equations.
Analyze the ODEs analytically or numerically, or run/animate stochastic
simulations on networks/well-mixed systems.

-  repository: https://github.com/benmaier/epipack/
-  documentation: http://epipack.benmaier.org/

.. code:: python

   import epipack as epk
   from epipack.vis import visualize
   import netwulf as nw

   network, _, __ = nw.load('cookbook/readme_vis/MHRN.json')
   N = len(network['nodes'])
   links = [ (l['source'], l['target'], 1.0) for l in network['links'] ]

   S, I, R = list("SIR")
   model = epk.StochasticEpiModel([S,I,R],N,links)\
               .set_link_transmission_processes([ (I, S, 1.0, I, I) ])\
               .set_node_transition_processes([ (I, 1.0, R) ])\
               .set_random_initial_conditions({ S: N-5, I: 5 })

   visualize(model, network, sampling_dt=0.1)

.. image:: https://github.com/benmaier/epipack/raw/master/img/SIR_example.gif
   :alt: sir-example

Idea
----

Simple compartmental models of infectious diseases are useful to
investigate effects of certain processes on disease dissemination. Using
pen and paper, quickly adding/removing compartments and transition
processes is easy, yet the analytical and numerical analysis or
stochastic simulations can be tedious to set up and debug—especially
when the model changes (even slightly). ``epipack`` aims at streamlining
this process such that all the analysis steps can be performed in an
efficient manner, simply by defining processes based on reaction
equations. ``epipack`` provides three main base classes to accomodate
different problems.

-  ``EpiModel``: Define a model based on transition, birth, death,
   fission, fusion, or transmission reactions, integrate the ordinary
   differential equations (ODEs) of the corresponding well-mixed system
   numerically or simulate it using Gillespie's algorithm. Process rates
   can be numerical functions of time and the system state.
-  ``SymbolicEpiModel``: Define a model based on transition, birth,
   death, fission, fusion, or transmission reactions. Obtain the ODEs,
   fixed points, Jacobian, and the Jacobian's eigenvalues at fixed
   points as symbolic expressions. Process rates can be symbolic
   expressions of time and the system state. Set numerical parameter
   values and integrate the ODEs numerically or simulate the stochastic
   systems using Gillespie's algorithm.
-  ``StochasticEpiModel``: Define a model based on node transition and
   link transmission reactions. Add conditional link transmission
   reactions. Simulate your model on any (un-/)directed, (un-/)weighted
   static/temporal network, or in a well-mixed system.

Additionally, epipack provides a visualization framework to animate
stochastic simulations on networks, lattices, well-mixed systems, or
reaction-diffusion systems based on ``MatrixEpiModel``.

Check out the `Example <#examples>`__ section for some demos.

Note that the internal simulation algorithm for network simulations is
based on the following paper:

"Efficient sampling of spreading processes on complex networks using a
composition and rejection algorithm", G.St-Onge, J.-G. Young, L.
Hébert-Dufresne, and L. J. Dubé, Comput. Phys. Commun. 240, 30-37
(2019), http://arxiv.org/abs/1808.05859.

Install
-------

.. code:: bash

   pip install epipack

``epipack`` was developed and tested for

-  Python 3.6
-  Python 3.7
-  Python 3.8

So far, the package's functionality was tested on Mac OS X and CentOS
only.

Dependencies
------------

``epipack`` directly depends on the following packages which will be
installed by ``pip`` during the installation process

-  ``numpy>=1.17``
-  ``scipy>=1.3``
-  ``sympy==1.6``
-  ``pyglet<1.6``
-  ``matplotlib>=3.0.0``
-  ``bfmplot>=0.0.7``
-  ``ipython>=7.14.0``
-  ``ipywidgets>=7.5.1``

Please note that **fast network simulations are only available if you
install**

-  ``SamplableSet==2.0``
   (`SamplableSet <http://github.com/gstonge/SamplableSet>`__)

**manually** (pip won't do it for you).

Documentation
-------------

The full documentation is available at
`epipack.benmaier.org <http://epipack.benmaier.org>`__.

Changelog
---------

Changes are logged in a `separate
file <https://github.com/benmaier/epipack/blob/master/CHANGELOG.md>`__.

License
-------

This project is licensed under the `MIT
License <https://github.com/benmaier/epipack/blob/master/LICENSE>`__.
Note that this excludes any images/pictures/figures shown here or in the
documentation.

Contributing
------------

If you want to contribute to this project, please make sure to read the
`code of
conduct <https://github.com/benmaier/epipack/blob/master/CODE_OF_CONDUCT.md>`__
and the `contributing
guidelines <https://github.com/benmaier/epipack/blob/master/CONTRIBUTING.md>`__.
In case you're wondering about what to contribute, we're always
collecting ideas of what we want to implement next in the `outlook
notes <https://github.com/benmaier/epipack/blob/master/OUTLOOK.md>`__.

|Contributor Covenant|

Examples
--------

Let's define an SIRS model with infection rate ``eta``, recovery rate
``rho``, and waning immunity rate ``omega`` and analyze the system

Pure Numeric Models
~~~~~~~~~~~~~~~~~~~

Basic Definition (EpiModel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define a pure numeric model with ``EpiModel``. Integrate the ODEs or
simulate the system stochastically.

.. code:: python

   from epipack import EpiModel
   import matplotlib.pyplot as plt
   import numpy as np

   S, I, R = list("SIR")
   N = 1000

   SIRS = EpiModel([S,I,R],N)\
       .set_processes([
           #### transmission process ####
           # S + I (eta=2.5/d)-> I + I
           (S, I, 2.5, I, I),

           #### transition processes ####
           # I (rho=1/d)-> R
           # R (omega=1/14d)-> S
           (I, 1, R),
           (R, 1/14, S),
       ])\
       .set_initial_conditions({S:N-10, I:10})

   t = np.linspace(0,40,1000) 
   result_int = SIRS.integrate(t)
   t_sim, result_sim = SIRS.simulate(t[-1])

   for C in SIRS.compartments:
       plt.plot(t, result_int[C])
       plt.plot(t_sim, result_sim[C])

.. image:: https://github.com/benmaier/epipack/raw/master/img/numeric_model.png
   :alt: numeric-model

Functional Rates
^^^^^^^^^^^^^^^^

It's also straight-forward to define temporally varying (functional)
rates.

.. code:: python

   import numpy as np
   from epipack import SISModel

   N = 100
   recovery_rate = 1.0

   def infection_rate(t, y, *args, **kwargs):
       return 3 + np.sin(2*np.pi*t/100)

   SIS = SISModel(
               infection_rate=infection_rate, 
               recovery_rate=recovery_rate,
               initial_population_size=N
               )\
           .set_initial_conditions({
               'S': 90,
               'I': 10,
           })

   t = np.arange(200)
   result_int = SIS.integrate(t)
   t_sim, result_sim = SIS.simulate(199)

   for C in SIS.compartments:
       plt.plot(t_sim, result_sim[C])
       plt.plot(t, result_int[C])

.. image:: https://github.com/benmaier/epipack/raw/master/img/numeric_model_time_varying_rate.png
   :alt: numeric-model-time-varying

Symbolic Models
~~~~~~~~~~~~~~~

Basic Definition
^^^^^^^^^^^^^^^^

Symbolic models are more powerful because they can do the same as the
pure numeric models while also offering the possibility to do analytical
evaluations

.. code:: python

   from epipack import SymbolicEpiModel
   import sympy as sy

   S, I, R, eta, rho, omega = sy.symbols("S I R eta rho omega")

   SIRS = SymbolicEpiModel([S,I,R])\
       .set_processes([
           (S, I, eta, I, I),
           (I, rho, R),
           (R, omega, S),
       ])    

Analytical Evaluations
^^^^^^^^^^^^^^^^^^^^^^

Print the ODE system in a Jupyter notebook

.. code:: python

   >>> SIRS.ODEs_jupyter()

.. image:: https://github.com/benmaier/epipack/raw/master/img/ODEs.png
   :alt: ODEs

Get the Jacobian

.. code:: python

   >>> SIRS.jacobian()

.. image:: https://github.com/benmaier/epipack/raw/master/img/jacobian.png
   :alt: Jacobian

Find the fixed points

.. code:: python

   >>> SIRS.find_fixed_points()

.. image:: https://github.com/benmaier/epipack/raw/master/img/fixed_points.png
   :alt: fixedpoints

Get the eigenvalues at the disease-free state in order to find the
epidemic threshold

.. code:: python

   >>> SIRS.get_eigenvalues_at_disease_free_state()
   {-omega: 1, eta - rho: 1, 0: 1}

Numerical Evaluations
^^^^^^^^^^^^^^^^^^^^^

Set numerical parameter values and integrate the ODEs numerically

.. code:: python

   >>> SIRS.set_parameter_values({eta: 2.5, rho: 1.0, omega:1/14})
   >>> t = np.linspace(0,40,1000)
   >>> result = SIRS.integrate(t)

If set up as

.. code:: python

   >>> N = 10000
   >>> SIRS = SymbolicEpiModel([S,I,R],N)

the system can simulated directly.

.. code:: python

   >>> t_sim, result_sim = SIRS.simulate(40)

Temporally Varying Rates
^^^^^^^^^^^^^^^^^^^^^^^^

Let's set up some temporally varying rates

.. code:: python

   from epipack import SymbolicEpiModel
   import sympy as sy

   S, I, R, eta, rho, omega, t, T = \
           sy.symbols("S I R eta rho omega t T")

   N = 1000
   SIRS = SymbolicEpiModel([S,I,R],N)\
       .set_processes([
           (S, I, 2+sy.cos(2*sy.pi*t/T), I, I),
           (I, rho, R),
           (R, omega, S),
       ])  

   SIRS.ODEs_jupyter()

.. image:: https://github.com/benmaier/epipack/raw/master/img/SIRS-forced-ODEs.png
   :alt: SIRS-forced-ODEs

Now we can integrate the ODEs or simulate the system using Gillespie's
SSA for inhomogeneous Poisson processes.

.. code:: python

   import numpy as np

   SIRS.set_parameter_values({
       rho : 1,
       omega : 1/14,
       T : 100,
   })
   SIRS.set_initial_conditions({S:N-100, I:100})
   _t = np.linspace(0,200,1000)
   result = SIRS.integrate(_t)
   t_sim, result_sim = SIRS.simulate(max(_t))

.. image:: https://github.com/benmaier/epipack/raw/master/img/symbolic_model_time_varying_rate.png
   :alt: SIRS-forced-results

Interactive Analyses
^^^^^^^^^^^^^^^^^^^^

``epipack`` offers a classs called ``InteractiveIntegrator`` that allows
an interactive exploration of a system in a Jupyter notebook.

Make sure to first run

.. code:: bash

   %matplotlib widget

in a cell.

.. code:: python

   from epipack import SymbolicEpiModel
   from epipack.interactive import InteractiveIntegrator, Range, LogRange
   import sympy

   S, I, R, R0, tau, omega = sympy.symbols("S I R R_0 tau omega")

   I0 = 0.01
   model = SymbolicEpiModel([S,I,R])\
                .set_processes([
                       (S, I, R0/tau, I, I),
                       (I, 1/tau, R),
                       (R, omega, S),
                   ])\
                .set_initial_conditions({S:1-I0, I:I0})

   # define a log slider, a linear slider and a constant value
   parameters = {
       R0: LogRange(min=0.1,max=10,step_count=1000),
       tau: Range(min=0.1,max=10,value=8.0),
       omega: 1/14
   }

   t = np.logspace(-3,2,1000)
   InteractiveIntegrator(model, parameters, t, figsize=(4,4))

.. image:: https://github.com/benmaier/epipack/raw/master/img/interactive.gif
   :alt: interactive

Pure Stochastic Models
~~~~~~~~~~~~~~~~~~~~~~

On a Network
^^^^^^^^^^^^

Let's simulate an SIRS system on a random graph (using the parameter
definitions above).

.. code:: python

   from epipack import StochasticEpiModel
   import networkx as nx

   k0 = 50
   R0 = 2.5
   rho = 1
   eta = R0 * rho / k0
   omega = 1/14
   N = int(1e4)
   edges = [ (e[0], e[1], 1.0) for e in \
             nx.fast_gnp_random_graph(N,k0/(N-1)).edges() ]

   SIRS = StochasticEpiModel(
               compartments=list('SIR'),
               N=N,
               edge_weight_tuples=edges
               )\
           .set_link_transmission_processes([
               ('I', 'S', eta, 'I', 'I'),
           ])\
           .set_node_transition_processes([
               ('I', rho, 'R'),
               ('R', omega, 'S'),
           ])\        
           .set_random_initial_conditions({
                                           'S': N-100,
                                           'I': 100
                                          })
   t_s, result_s = SIRS.simulate(40)

.. image:: https://github.com/benmaier/epipack/raw/master/img/network_simulation.png
   :alt: network-simulation

Visualize
^^^^^^^^^

Likewise, it's straight-forward to visualize this system

.. code:: python

   >>> from epipack.vis import visualize
   >>> from epipack.networks import get_random_layout
   >>> layouted_network = get_random_layout(N, edges)
   >>> visualize(SIRS, layouted_network, sampling_dt=0.1, config={'draw_links': False})

.. image:: https://github.com/benmaier/epipack/raw/master/img/SIRS_visualization.gif
   :alt: sirs-example

On a Lattice
^^^^^^^^^^^^

A lattice is nothing but a network, we can use ``get_grid_layout`` and
``get_2D_lattice_links`` to set up a visualization.

.. code:: python

   from epipack.vis import visualize
   from epipack import (
       StochasticSIRModel, 
       get_2D_lattice_links, 
       get_grid_layout
   )

   # define links and network layout
   N_side = 100
   N = N_side**2
   links = get_2D_lattice_links(N_side, periodic=True, diagonal_links=True)
   lattice = get_grid_layout(N)

   # define model
   R0 = 3; recovery_rate = 1/8
   model = StochasticSIRModel(N,R0,recovery_rate,
                              edge_weight_tuples=links)
   model.set_random_initial_conditions({'I':20,'S':N-20})

   sampling_dt = 1

   visualize(model,lattice,sampling_dt,
           config={
                    'draw_nodes_as_rectangles':True,
                    'draw_links':False,
                  }
             )

.. image:: https://github.com/benmaier/epipack/raw/master/img/SIR_lattice_vis.gif
   :alt: sir-lattice

Reaction-Diffusion Models
~~~~~~~~~~~~~~~~~~~~~~~~~

Since reaction-diffusion systems in discrete space can be interpreted as
being based on reaction equations, we can set those up using
``epipack``'s framework.

Checkout the docs on `Reaction-Diffusion
Systems <http://epipack.benmaier.org/tutorial/reaction_diffusion.html>`__.

Every node in a network is associated with a compartment and we're using
``MatrixEpiModel`` because it's faster than ``EpiModel``.

.. code:: python

   from epipack import MatrixEpiModel

   N = 100
   base_compartments = list("SIR")
   compartments = [ (node, C) for node in range(N) for C in base_compartments ]
   model = MatrixEpiModel(compartments)

Now, we define both epidemiological and movement processes on a
hypothetical list ``links``.

.. code:: python

   infection_rate = 2
   recovery_rate = 1
   mobility_rate = 0.1

   quadratic_processes = []
   linear_processes = []

   for node in range(N):
       quadratic_processes.append(
               (  (node, "S"), (node, "I"), infection_rate, (node, "I"), (node, "I") ),
           )

       linear_processes.append(
                 ( (node, "I"), recovery_rate, (node, "R") ) 
           )

   for u, v, w in links:
       for C in base_compartments:

           linear_processes.extend([
                     ( (u, C), w*mobility_rate, (v, C) ),
                     ( (v, C), w*mobility_rate, (u, C) ),
               ])

.. image:: https://github.com/benmaier/epipack/raw/master/img/reac_diff_lattice.gif
   :alt: reac-diff-lattice

Dev notes
---------

Fork this repository, clone it, and install it in dev mode.

.. code:: bash

   git clone git@github.com:YOURUSERNAME/epipack.git
   make

If you want to upload to PyPI, first convert the new ``README.md`` to
``README.rst``

.. code:: bash

   make readme

It will give you warnings about bad ``.rst``-syntax. Fix those errors in
``README.rst``. Then wrap the whole thing

.. code:: bash

   make pypi

It will probably give you more warnings about ``.rst``-syntax. Fix those
until the warnings disappear. Then do

.. code:: bash

   make upload

.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg
   :target: code-of-conduct.md
