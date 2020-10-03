|logo|

epipack
=======

Fast prototyping of epidemiological models based on reaction equations.
Analyze the ODEs analytically or numerically, or run/animate stochastic
simulations on networks/well-mixed systems.

-  repository: https://github.com/benmaier/epipack/
-  documentation: https://epipack.readthedocs.io/

.. code:: python

   import epipack as epk
   from epipack.vis import visualize
   import netwulf as nw

   network, _, __ = nw.load('cookbook/readme_vis/MHRN.json')
   N = len(network['nodes'])
   links = [ (l['source'], l['target'], 1.0) for l in network['links'] ]

   model = epk.StochasticEpiModel(["S","I","R"],N,links)\
               .set_link_transmission_processes([ ("I", "S", 1.0, "I", "I") ])\
               .set_node_transition_processes([ ("I", 1.0, "R") ])\
               .set_random_initial_conditions({ "S": N-5, "I": 5 })

   visualize(model, network, sampling_dt=0.1)

|sir-example|

Idea
----

Simple compartmental models of infectious diseases are useful to
investigate effects of certain processes on disease dissemination. Using
pen and paper, quickly adding/removing compartments and transition
processes is easy, yet the analytical and numerical analysis or
stochastic simulations can be tedious to set up and debug—especially
when the model changes (even slightly). ``epipack`` aims to streamline
this process such that all the analysis steps can be performed in an
efficient manner, simply by defining processes based on reaction
equations. ``epipack`` provides three base classes to accomodate
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
   static network, or in a well-mixed system.

Additionally, epipack provides a visualization framework to animate
stochastic simulations on networks, lattices, or well-mixed systems.

Check out the `Example <#examples>`__ section for some demos.

Install
-------

.. code:: bash

   git clone git@github.com:benmaier/epipack.git
   pip install ./epipack

``epipack`` was developed and tested for

-  Python 3.7

So far, the package's functionality was tested on Mac OS X only.

Dependencies
------------

``epipack`` directly depends on the following packages which will be
installed by ``pip`` during the installation process

-  ``numpy>=1.17``
-  ``scipy>=1.3``
-  ``sympy==1.6``
-  ``pyglet<1.6``
-  ``ipython>=7.17.0``
-  ``ipywidgets>=7.5.1``

Please note that **fast network simulations are only available if you
install**

-  ``SamplableSet==2.0``
   (`SamplableSet <http://github.com/gstonge/SamplableSet>`__)

**manually** (pip won't do it for you).

Documentation
-------------

The full documentation is available at epipack.benmaier.org.

Changelog
---------

Changes are logged in a `separate
file <https://github.com/benmaier/epipack/blob/master/CHANGELOG.md>`__.

License
-------

This project is licensed under the `MIT
License <https://github.com/benmaier/epipack/blob/master/LICENSE>`__.

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

   for C in model.compartments:
       plt.plot(t, result_int[C])
       plt.plot(t_sim, result_sim[C])

|numeric-model|

It's also straight-forward to define temporally varying rates.

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

|numeric-model-time-varying|

Symbolic Models
~~~~~~~~~~~~~~~

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

Print the ODE system in a Jupyter notebook

.. code:: python

   >>> SIRS.ODEs_jupyter()

|ODEs|

Get the Jacobian

.. code:: python

   >>> SIRS.jacobian()

|Jacobian|

Find the fixed points

.. code:: python

   >>> SIRS.find_fixed_points()

|fixedpoints|

Get the eigenvalues at the disease-free state in order to find the
epidemic threshold

.. code:: python

   >>> SIRS.get_eigenvalues_at_disease_free_state()
   {-omega: 1, eta - rho: 1, 0: 1}

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

|SIRS-forced-ODEs|

Let's integrate/simulate these equations

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

|SIRS-forced-results|

Pure Stochastic Models
~~~~~~~~~~~~~~~~~~~~~~

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

|network-simulation|

Likewise, it's straight-forward to visualize this system

.. code:: python

   >>> from epipack.vis import visualize
   >>> from epipack.networks import get_random_layout
   >>> layouted_network = get_random_layout(N, edges)
   >>> visualize(SIRS, layouted_network, sampling_dt=0.1, config={'draw_links': False})

|sirs-example|

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

.. |logo| image:: https://github.com/benmaier/epipack/raw/master/img/logo_flatter_medium.png
.. |sir-example| image:: https://github.com/benmaier/epipack/raw/master/img/SIR_example.gif
.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg
   :target: code-of-conduct.md
.. |numeric-model| image:: https://github.com/benmaier/epipack/raw/master/img/numeric_model.png
.. |numeric-model-time-varying| image:: https://github.com/benmaier/epipack/raw/master/img/numeric_model_time_varying_rate.png
.. |ODEs| image:: https://github.com/benmaier/epipack/raw/master/img/ODEs.png
.. |Jacobian| image:: https://github.com/benmaier/epipack/raw/master/img/jacobian.png
.. |fixedpoints| image:: https://github.com/benmaier/epipack/raw/master/img/fixed_points.png
.. |SIRS-forced-ODEs| image:: https://github.com/benmaier/epipack/raw/master/img/SIRS-forced-ODEs.png
.. |SIRS-forced-results| image:: https://github.com/benmaier/epipack/raw/master/img/symbolic_model_time_varying_rate.png
.. |network-simulation| image:: https://github.com/benmaier/epipack/raw/master/img/network_simulation.png
.. |sirs-example| image:: https://github.com/benmaier/epipack/raw/master/img/SIRS_visualization.gif
