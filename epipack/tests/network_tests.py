import unittest

import numpy as np
import sympy

from epipack.networks import (
            get_2D_lattice_links
        )

_xor = lambda a, b: (a and not b) or (not a and b)

class NetworkModelsTest(unittest.TestCase):

    def test_unperiodic_undiagonal(self):

        links = get_2D_lattice_links(2)

        expected_links = [
                    (0,1,1.0),
                    (0,2,1.0),
                    (1,3,1.0),
                    (2,3,1.0),
                ]

        assert(set(links) == set(expected_links))

    def test_periodic_undiagonal(self):

        links = get_2D_lattice_links(2,periodic=True)

        expected_links = [
                    (0,1,1.0),
                    (0,2,1.0),
                    (1,3,1.0),
                    (2,3,1.0),
                ]

        assert(set(links) == set(expected_links))

    def test_unperiodic_diagonal(self):

        links = get_2D_lattice_links(2,diagonal_links=True)

        expected_links = [
                    (0,1,1.0),
                    (0,2,1.0),
                    (0,3,1.0),
                    (1,3,1.0),
                    (1,2,1.0),
                    (2,3,1.0),
                ]

        assert(set(links) == set(expected_links))

    def test_periodic_diagonal(self):

        links = get_2D_lattice_links(2,periodic=True,diagonal_links=True)

        expected_links = [
                    (0,1,1.0),
                    (0,2,1.0),
                    (0,3,1.0),
                    (1,3,1.0),
                    (1,2,1.0),
                    (2,3,1.0),
                ]

        assert(set(links) == set(expected_links))

if __name__ == "__main__":

    T = NetworkModelsTest()
    T.test_unperiodic_undiagonal()
    T.test_unperiodic_diagonal()
    T.test_periodic_undiagonal()
    T.test_periodic_diagonal()
