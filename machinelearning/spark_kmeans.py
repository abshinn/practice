#!/usr/bin/env spark-submit
"""
Implement KMeans using Lloyd's algorithm.
"""

from pyspark import SparkContext

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class KMeans(object):

    def fit(self, k_clusters):
        """find k clusters"""
        self.k_clusters = k_clusters

        self.m, self.n = X.shape
        self._initialize_centroids(X)

        iter = 0
        while self._has_converged(): 
            print "iter: {}".format(iter)
            iter += 1
            labels = self.foreach(self._assign_data_to_centroids)

#             self._update_centroids(X)

    def _initialize_centroids(self, X):
        """pick random data points as inital cluster centroids"""
        pass
#         self.old_label, self.label = None, np.zeros(self.m)
#         self.centroids = X[np.random.choice(range(self.m), self.n_clusters, replace=False),:]

    def _assign_data_to_centroids(self, u):
        """Assign data points to current centroids. To be applied to a Spark RDD foreach() method. 
        INPUT: u, row vector in X
        """
        return np.argmin( [np.linalg.norm(u - centroid) for centroid in self.centroids] )

    def _update_centroids(self, X):
        """update centroid positions based on current assignment"""
        pass
#         self.centroids = np.zeros((self.n_clusters, self.n))
#         for k in xrange(self.n_clusters):
#             self.centroids[k,:] = X[self.label == k,:].mean(axis=0)

    def _has_converged(self):
        pass
#         return not np.all(self.old_label == self.label)


def sample_set():
    x = np.linspace(0, 10)
    y = x + 3 * np.random.randn(50)
    X = np.c_[x, y]
    return X


if __name__ == "__main__":
    sc = SparkContext(appName="PythonKMeans")
    X = sc.parallelize(sample_set())
    print X
