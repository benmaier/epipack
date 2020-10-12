Testing
-------

All critical methods come with extensive test cases.

This includes tests based on the Kullback-Leibler divergence
between expected distributions and distributions obtained
by simulations on toy systems (including networks and
time-varying rate simulations).

Please make sure to run the entire test suite before
opening a pull request. This can be done by calling

.. code:: bash

    make test

Alternatively, the test command is

.. code:: bash

	pytest --cov=epipack epipack/tests/

The tests should take 15-20 minutes. After the tests
have passed, you can check the coverage by calling

.. code:: bash

    open htmlcov/index.html

(or ``xopen`` on Linux).

