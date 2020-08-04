
Here, :math:`X_n` represents a compartment :math:`X\in\{S,I,R\}` 
at location :math:`n` with population size :math:`N_n`,
and :math:`w_{nm}` represents an individual transition rate
of an individual from location `m` to location `n`.

The air traffic network shall be quantified by :math:`F_{nm}`,
which counts the number of passengers that flew 
from location `m` to location `n`
during a defined period of time.

Now, we have a problem, because usually we want the population
size per location to be constant. It has been shown that
the equations can be rescaled such that the emerging ODEs
describe concentrations instead of total counts. In this
picture, we have

.. math::

    S_n + I_n \stackrel{\longrightarrow}{\eta} 2I_n

    I_n \stackrel{\longrightarrow}{\rho} R_n

    X_m \stackrel{\longrightarrow}{\gamma P_{nm}} X_n

    P_{nm} = F_{nm} / \sum{\ell} F_{\ell m}

where :math:`\gamma` is a global mobility rate and 
:math:`P_{nm}` is the probability that a passenger
starting at location `m` directly flies to location `n`.


