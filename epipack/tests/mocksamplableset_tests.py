import unittest

import numpy as np

from epipack.mock_samplable_set import MockSamplableSet

class   MSSTest(unittest.TestCase):

    def test_all(self):

        s = MockSamplableSet(1.0,2.0,{0:1.2,3:1.8,})

        assert(all(i==j for i, j in zip([0,3],s.items)))
        assert(all(i==j for i, j in zip([1.2,1.8],s.weights)))
        assert(len(s) == 2)
        assert(s.total_weight() == 3.0)
    
        expected = [
            (3, 1.8),
            (3, 1.8),
            (0, 1.2),
            (0, 1.2),
            (0, 1.2),
            ]
        samples = []
        np.random.seed(1)
        for _ in range(5):
            a = s.sample()
            samples.append(a)

        assert(all([
                    i[0] == j[0] and i[1] == j[1] \
                    for i, j in zip(expected, samples)
                ]))

        a = s[0]
        b = s[0]
        assert(a==b)

        s[0] = 2
        assert(all(i==j for i, j in zip([0,3],s.items)))
        assert(all(i==j for i, j in zip([2.,1.8],s.weights)))
        assert(s.total_weight() == 3.8)

        s[1] = 1.3
        assert(all(i==j for i, j in zip([0,1,3],s.items)))
        assert(all(i==j for i, j in zip([2.,1.3,1.8],s.weights)))
        assert(s.total_weight() == 5.1)

        del s[3]
        assert(all(i==j for i, j in zip([0,1],s.items)))
        assert(all(i==j for i, j in zip([2.,1.3],s.weights)))
        assert(s.total_weight() == 3.3)

        exp = [
                (0, 2.0),
                (1, 1.3),
              ]
        samples = [ (item, weight) for item, weight in s ]
        assert(set(exp) == set(samples))

        assert(0 in s)
        assert(45 not in s)


if __name__ == "__main__":

    T = MSSTest()
    T.test_all()
