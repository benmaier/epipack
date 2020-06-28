
![logo](https://github.com/benmaier/epipack/raw/master/img/logo_medium.png)

## epipack

Fast prototyping of epidemiological models based on reaction equations. Analyze the ODEs analytically or numerically, or run stochastic simulations on networks/well-mixed systems.

    git clone git@github.com:benmaier/epipack.git
    pip install ./epipack

`epipack` was developed and tested for 

* Python 3.7

So far, the package's functionality was tested on Mac OS X only.

## Dependencies

`epipack` directly depends on the following packages which will be installed by `pip` during the installation process

* `numpy>=1.17`
* `scipy>=1.3`
* `sympy==1.6`

Please note that **fast network simulations are only available if you install** 

* `SamplableSet==2.0` ([SamplableSet](http://github.com/gstonge/SamplableSet))

**manually** (pip won't do it for you).

## Documentation

The full documentation is available at XXX.

## Examples

Let's define an SIRS model with infection rate `eta`, recovery rate `rho`, and waning immunity rate `omega` and ananalyze the system

### Symbolic evaluations

```python
from epipack import SymbolicEpiModel
import sympy as sy

S, I, R, eta, rho, omega = sy.symbols("S I R eta rho omega")

SIRS = SymbolicEpiModel([S,I,R])

SIRS.set_processes([
    #### transmission process ####
    # S + I (eta)-> I + I
    (S, I, eta, I, I),

    #### transition processes ####
    # I (rho)-> R
    # R (omega)-> S
    (I, rho, R),
    (R, omega, S),

])
```

Print the ODE system in a Jupyter notebook

```
>>> SIRS.ODEs_jupyter()
```

![ODEs](https://github.com/benmaier/epipack/raw/master/img/ODEs.png)

Get the Jacobian

```python
>>> SIRS.jacobian()
```

![Jacobian](https://github.com/benmaier/epipack/raw/master/img/jacobian.png){:height="50%" width="50%"}

Find the fixed points

```python
>>> SIRS.find_fixed_points()
```

![fixedpoints](https://github.com/benmaier/epipack/raw/master/img/fixed_points.png)

Get the eigenvalues at the disease-free state in order to find the epidemic threshold

```python
>>> SIRS.get_eigenvalues_at_disease_free_state()
{-omega: 1, eta - rho: 1, 0: 1}
```

### Numeric evaluations

In order to numerically integrate the ODEs, use `DeterministicEpiModel`

```python
from epipack import DeterministicEpiModel
S, I, R = list("SIR")
R0 = 2.5
rho = recovery_rate = 1 # let's say 1/days
eta = infection_rate = R0 * recovery_rate
omega = 1/14 # in units of 1/days

SIRS = DeterministicEpiModel([S,I,R])

SIRS.set_processes([
    #### transmission process ####
    # S + I (eta)-> I + I
    (S, I, eta, I, I),

    #### transition processes ####
    # I (rho)-> R
    # R (omega)-> S
    (I, rho, R),
    (R, omega, S),

])

SIRS.set_initial_conditions({S:1-0.01, I:0.01})

t = np.linspace(0,100,1000) 
result = SIRS.integrate(t)
```

![integrated-ODEs](https://github.com/benmaier/epipack/raw/master/img/integrated_ODEs.png)

### Stochastic simulations

Let's simulate the system on a random graph (using the parameter definitions above).

```python
from epipack import StochasticEpiModel
import networkx as nx

k0 = 50
eta = R0 * rho / k0
N = int(1e4)
edges = [ (e[0], e[1], 1.0) for e in nx.fast_gnp_random_graph(N,k0/(N-1)).edges() ]

SIRS = StochasticEpiModel([S,I,R],N,edge_weight_tuples=edges)

SIRS.set_link_transmission_processes([
    #### transmission process ####
    # I + S (eta)-> I + I
    (I, S, eta, I, I),
])

SIRS.set_node_transition_processes([
    #### transition processes ####
    # I (rho)-> R
    # R (omega)-> S
    (I, rho, R),
    (R, omega, S),

])

SIRS.set_random_initial_conditions({S:N-int(1e-2*N), I:int(1e-2*N)})
t_s, result_s = SIRS.simulate(100)
```
![integrated-ODEs](https://github.com/benmaier/epipack/raw/master/img/stochastic_simulation.png)

## Changelog

Changes are logged in a [separate file](https://github.com/benmaier/epipack/blob/master/CHANGELOG.md).

## License

This project is licensed under the [MIT License](https://github.com/benmaier/epipack/blob/master/LICENSE).

## Contributing

If you want to contribute to this project, please make sure to read the [code of conduct](https://github.com/benmaier/epipack/blob/master/CODE_OF_CONDUCT.md) and the [contributing guidelines](https://github.com/benmaier/epipack/blob/master/CONTRIBUTING.md). In case you're wondering about what to contribute, we're always collecting ideas of what we want to implement next in the [outlook notes](https://github.com/benmaier/epipack/blob/master/OUTLOOK.md).

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](code-of-conduct.md)

## Dev notes

Fork this repository, clone it, and install it in dev mode.

```bash
git clone git@github.com:YOURUSERNAME/epipack.git
make
```

If you want to upload to PyPI, first convert the new `README.md` to `README.rst`

```bash
make readme
```

It will give you warnings about bad `.rst`-syntax. Fix those errors in `README.rst`. Then wrap the whole thing 

```bash
make pypi
```

It will probably give you more warnings about `.rst`-syntax. Fix those until the warnings disappear. Then do

```bash
make upload
```
