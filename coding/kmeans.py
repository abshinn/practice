#!/usr/bin/env python2.7 -B -tt
"""
Implement Kmeans using Lloyd's algorithm.
"""

import numpy as np
np.random

class KMeans(object):

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def _initialize_centroids(self, X):
        """pick random data points as inital cluster centroids"""
        self.centroids = X[np.random.choice(range(self.m), self.n_clusters, replace=False),:]

    def _assign_centroids(self, X):
        """assign data points to current centroids"""
        pass

    def _update_centroids(self):
        """update centroid positions based on current assignment"""
        pass

    def fit(self, X):
        self.m, self.n = X.shape

if __name__ == "__main__":
    pass
