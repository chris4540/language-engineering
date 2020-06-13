import numpy as np
from sklearn.neighbors import NearestNeighbors
samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
neigh.fit(samples)

neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)

nbrs = neigh.radius_neighbors([[0, 0, 1.3]], 0.4, return_distance=False)
np.asarray(nbrs[0][0])
