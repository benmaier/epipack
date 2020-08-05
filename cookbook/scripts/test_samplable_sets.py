from epipack.mock_samplable_set import MockSamplableSet
import numpy as np
from SamplableSet import SamplableSet

from time import time

N = int(2e4)
n = np.arange(N)
np.random.shuffle(n)
weights = np.random.random(N) + 0.1
_min, _max = np.amin(weights), np.amax(weights)

weighted_items = [ (int(_n), float(_w)) for _n, _w in zip(n, weights)]


print("#========= CREATING ========")
start = time()
smpset = SamplableSet(0.1, 1.1, weighted_items)
end = time()

print("Orig:", end-start,"seconds")

start = time()
mckset = MockSamplableSet(0.1, 1.1, weighted_items)
end = time()

print("Mock:", end-start,"seconds")

print("#========= ADDING ========")
n = np.arange(N) + N
np.random.shuffle(n)
weights = np.random.random(N) + 0.1

start = time()
for _n, w in zip(n,weights):
    smpset[int(_n)] = w
end = time()

print("Orig:", end-start,"seconds")

start = time()
for _n, w in zip(n,weights):
    mckset[_n] = w
end = time()

print("Mock:", end-start,"seconds")

print("#========= SAMPLING ========")
N_samples = int(1e4)

start = time()
for smpl in range(N_samples):
    smpset.sample()
end = time()

print("Orig:", end-start,"seconds")

start = time()
for smpl in range(N_samples):
    mckset.sample()
end = time()

print("Mock:", end-start,"seconds")

print("#========= DELETING ========")
N_samples = int(1e4)

start = time()
for _n in n:
    del smpset[_n]
end = time()

print("Orig:", end-start,"seconds")

start = time()
for _n in n:
    del mckset[_n]
end = time()

print("Mock:", end-start,"seconds")
