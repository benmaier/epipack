"""
Contains a MockSamplableSet class that mimicks the behavior of 
github.com/gstonge/SamplableSet but is less efficient.
"""

import numpy as np

class MockSamplableSet:

    def __init__(self,min_weight,max_weight,weighted_elements=[],cpp_type='int'):

        self.min_weight = min_weight
        self.max_weight = max_weight

        if type(weighted_elements) == dict:
            weighted_elements = list(weighted_elements.items())

        self.items = np.array([ e[0] for e in weighted_elements ],dtype=cpp_type)
        self.weights = np.array([ e[1] for e in weighted_elements ],dtype=float)
        sort_ndx = np.argsort(self.items)
        self.items = self.items[sort_ndx]
        self.weights = self.weights[sort_ndx]
        self._total_weight = self.weights.sum()

        if np.any(self.weights < self.min_weight):
            raise ValueError("There are weights below the limit.")

        if np.any(self.weights > self.max_weight):
            raise ValueError("There are weights above the limit.")

    def sample(self):
        ndx = np.random.choice(len(self.items),p=self.weights/self._total_weight)
        return self.items[ndx], self.weights[ndx]

    def __getitem__(self,key):
        found_key, ndx = self._find_key(key)
        if not found_key:
            raise KeyError("`",key,"` is not in this set.")
        else:
            return self.weights[ndx]

    def __delitem__(self,key):
        found_key, ndx = self._find_key(key)
        if found_key:
            self.items = np.delete(self.items, ndx)
            self.weights = np.delete(self.weights, ndx)
            self._total_weight = self.weights.sum()

    def __setitem__(self,key,value):
        if value < self.min_weight or value > self.max_weight:
            raise ValueError('Inserting element-weight pair ', key, value,
                             'has weight value out of bounds of ', self.min_weight, self.max_weight)
        found_key, ndx = self._find_key(key) 
        if not found_key:
            self.items = np.insert(self.items, ndx, key)
            self.weights = np.insert(self.weights, ndx, value)
        else:
            self.weights[ndx] = value
            
        self._total_weight = self.weights.sum()

    def _find_key(self,key):
        ndx = np.searchsorted(self.items, key)
        return ( not ((ndx == len(self.items) or self.items[ndx] != key)), ndx )

    def __iter__(self):
        self._ndx = 0
        return self

    def __next__(self):
        if self._ndx < len(self.items):
            i, w = self.items[self._ndx], self.weights[self._ndx]            
            self._ndx += 1
            return (i,w)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.items)

    def __contains__(self,key):
        return self._find_key(key) is not None

    def total_weight(self):
        return self._total_weight



if __name__ == "__main__":

    s = MockSamplableSet(1.0,2.0,{0:1.2,3:1.8,})

    print(s.items)
    print(s.weights)
    print(len(s))
    print(s.total_weight())

    for _ in range(5):
        print(s.sample())


    print(s[0])
    print(s[0])

    s[0] = 2
    print(s.items)
    print(s.weights)
    print(s.total_weight())

    s[1] = 1.3
    print(s.items)
    print(s.weights)
    print(s.total_weight())

    del s[3]
    print(s.items)
    print(s.weights)
    print(s.total_weight())

    for item, weight in s:
        print(item, weight)

    print(0 in s)
