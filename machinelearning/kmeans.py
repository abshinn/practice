#!/usr/bin/env python2.7 -B -tt
""" Implement Kmeans using Lloyd's algorithm. """

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class KMeans(object):

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def _initialize_centroids(self, X):
        """ Pick random data points as inital cluster centroids. """

        self.old_label, self.label = None, np.zeros(self.m)
        self.centroids = X[np.random.choice(range(self.m), self.n_clusters, replace=False),:]

    def _assign_data_to_centroids(self, X):
        """ Assign data points to current centroids. """

        self.old_label, self.label = self.label, np.zeros(self.m)

        for i, example in enumerate(X):
            self.label[i] = np.argmin( [np.linalg.norm(example - centroid) for centroid in self.centroids] )

    def _update_centroids(self, X):
        """ Update centroid positions based on current assignment. """

        self.centroids = np.zeros((self.n_clusters, self.n))
        for k in xrange(self.n_clusters):
            self.centroids[k,:] = X[self.label == k,:].mean(axis=0)

    def _has_converged(self):
        return not np.all(self.old_label == self.label)

    def plot_centroids(self, p):
        """ Plot centroids for a two-dimensional feature space. """

        for cluster in self.centroids: 
            p.plot(cluster[0], cluster[1], marker="x", markerfacecolor="red", markeredgewidth=4, markersize=10, alpha=0.7)

    def plot_residuals(self, p, X):
        """ Plot residual from data point to corresponding centroid for a two-dimensional feature space. """

        for k, centroid in enumerate(self.centroids):
            U = X[self.label == k,:]
            for u in U:
                p.plot([u[0], centroid[0]], [u[1], centroid[1]], linewidth=2, color="black", alpha=0.1)

    def fit(self, X):
        """ Find k clusters.

        INPUT:
            X -- numpy.ndarray feature matrix
        """
        self.m, self.n = X.shape
        self._initialize_centroids(X)

        iter = 0
        while self._has_converged(): 
            print "iter: {}".format(iter)
            iter += 1
            self._assign_data_to_centroids(X)
            self._update_centroids(X)


def sample_set():
    """ Generate positively correlated non-seperable data. """

    x = np.linspace(0, 10)
    y = x + 3 * np.random.randn(50)
    X = np.c_[x, y]
    return X


def example():
    """ Example fitting 3 clusters to non-separable data. """

    X = sample_set()
    plt.scatter(X[:,0], X[:,1], marker="o", color="steelblue")

    km = KMeans(n_clusters=3)
    km.fit(X)

    km.plot_centroids(plt)
    km.plot_residuals(plt, X)
    plt.show()


if __name__ == "__main__":
    example()

