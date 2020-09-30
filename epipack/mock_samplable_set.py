"""
Contains a MockSamplableSet class that mimicks the behavior of 
github.com/gstonge/SamplableSet but is less efficient.
"""

import numpy as np

def choice(arr,p):
    """
    Returns a random element from ``arr`` with probability given in array ``p``.
    If ``arr`` is not an iterable, the function returns the index of the chosen element.
    """
    ndx = np.argmax(np.random.rand()<np.cumsum(p))
    try:
        return arr[ndx]
    except TypeError as e:
        return ndx


class MockSamplableSet:
    """
    A set of items that can be sampled with probability
    proportional to a corresponding item weight.

    Mimicks the behavior of github.com/gstonge/SamplableSet
    without being as efficient.

    Works similar to Python's set, with ``__getitem__``, 
    ``__setitem__``, ``__delitem__``, ``__iter__``,
    ``__len__``, ``__contains__``.

    Parameters
    ==========
    min_weight : float
        minimum possible weight
    max_weight : float
        maximum possible weight
    weighted_elements : list, default = []
        list of 2-tuples, first entry an item, second entry a weight
    cpp_type : str, default = 'int'
        The type of the items.

    Attributes
    ==========
    min_weight : float
        minimum possible weight
    max_weight : float
        maximum possible weight
    items : numpy.ndarray
        list of items in this set
    weights : numpy.ndarray
        list of corresponding weights
    """

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
        """
        Random sample from the set, sampled
        with probability proportional to items' weight.

        Returns
        =======
        item : cpp_type
            An item from the set
        weight : float
            The weight of the item
        """

        #ndx = np.random.choice(len(self.items),p=self.weights/self._total_weight)
        #ndx = np.argwhere(np.random.rand()<np.cumsum(self.weights/self._total_weight))[0][0]
        ndx = choice(len(self.items),p=self.weights/self._total_weight)
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
            raise ValueError('Inserting element-weight pair ' + str(key) +" "+ str(value)+" \n" +\
                             'has weight value out of bounds of ' + str(self.min_weight) + " " + \
                             str(self.max_weight))
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
        return self._find_key(key)[0]

    def total_weight(self):
        """Obtain the total weight of the set"""
        return self._total_weight

    def clear(self):
        """Reset the set. Not implemented yet."""
        pass



if __name__ == "__main__":    # pragma: no cover

    s = MockSamplableSet(1.0,2.0,{0:1.2,3:1.8,})

    print('===========')
    print(s.items)
    print(s.weights)
    print(len(s))
    print(s.total_weight())

    np.random.seed(1)
    print('===========')
    for _ in range(5):
        print(s.sample())


    print('===========')
    print(s[0])
    print(s[0])

    s[0] = 2
    print('===========')
    print(s.items)
    print(s.weights)
    print(s.total_weight())

    s[1] = 1.3
    print('===========')
    print(s.items)
    print(s.weights)
    print(s.total_weight())

    del s[3]
    print('===========')
    print(s.items)
    print(s.weights)
    print(s.total_weight())

    print('===========')
    for item, weight in s:
        print(item, weight)

    print('===========')
    print(0 in s)
    print(45 in s)

