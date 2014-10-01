#!/usr/bin/env python2.7 -B -tt
"""
Implement Kmeans using Lloyd's algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class KMeans(object):

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def _initialize_centroids(self, X):
        """pick random data points as inital cluster centroids"""
        self.centroids = X[np.random.choice(range(self.m), self.n_clusters, replace=False),:]

    def _assign_data_to_centroids(self, X):
        """assign data points to current centroids"""

        self.label = np.zeros((self.m, self.n_clusters))

        for i, example in enumerate(X):
            self.label[i] = np.argmin( [np.linalg.norm(example - centroid) for centroid in self.centroids] )

    def _update_centroids(self):
        """update centroid positions based on current assignment"""
        pass

    def _has_converged(self):
        return True

    def plot_centroids(self, p):
        for cluster in self.centroids: 
            print cluster[0], cluster[1]
            p.plot(cluster[0], cluster[1], marker="x", markerfacecolor="red", markeredgewidth=4, markersize=10, alpha=.7)

    def fit(self, X):
        """
        INPUT: X -- numpy.ndarray feature matrix
        """
        self.m, self.n = X.shape
        self._initialize_centroids(X)
        self._assign_data_to_centroids(X)

#         while self._has_converged(): 
#             pass


def sample_set():
    x = np.linspace(0, 10)
    y = x + 3 * np.random.randn(50)
    X = np.c_[x, y]
    return X


if __name__ == "__main__":
    X = sample_set()
    plt.scatter(X[:,0], X[:,1], marker="o", color="steelblue")

    km = KMeans(n_clusters=3)
    km.fit(X)
    km.plot_centroids(plt)

    plt.show()
