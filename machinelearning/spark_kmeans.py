#!/usr/bin/env spark-submit
"""
Implement KMeans using Lloyd's algorithm in Spark.
"""

from pyspark import SparkContext, SparkConf

from operator import sub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class KMeans(object):

    def __init__(self, k_clusters):
        self.k_clusters = k_clusters

    def fit(self, Xrdd):
        """find k clusters"""

        self.centroids = Xrdd.takeSample(False, self.k_clusters)
        for centroid in self.centroids:
            print centroid

        old_labels = None
        labels = Xrdd.map(self._assign_data_to_centroids)

#         iter = 0
#         while self._has_converged(): 
#             print "iter: {}".format(iter)
#             iter += 1
#             labels = Xrdd.foreach(self._assign_data_to_centroids)
# 
#             self._update_centroids(X)

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
    sc = SparkContext(appName="PythonKMeans", conf=SparkConf().set("spark.driver.host", "localhost"))
    X = sc.parallelize(sample_set())

    km = KMeans(k_clusters=3)

    km.fit(X)

