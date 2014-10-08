#!/usr/bin/env spark-submit
"""
Implement KMeans using Lloyd's algorithm in Spark.
"""

from pyspark import SparkContext, SparkConf
import numpy as np
np.random.seed(42)

class KMeans(object):

    def __init__(self, k_clusters):
        self.k_clusters = k_clusters

    def fit(self, Xrdd):
        """find k clusters"""

        self.centroids = Xrdd.takeSample(False, self.k_clusters, 42)

        for centroid in self.centroids:
            print centroid

        label = Xrdd.map(self._assign_data_to_centroids)
        Xrdd = label.zip(Xrdd)

        prev_dist = None
        current_dist = Xrdd.map(lambda (l, u): np.linalg.norm(u - self.centroids[l])).sum()

        iter = 0
        while current_dist != prev_dist: 
            print "iter: {}".format(iter)
            iter += 1

            self.centroids = self._update_centroids(Xrdd)

            Xrdd = Xrdd.map(lambda (l, u): (self._assign_data_to_centroids(u), u))

            for centroid in self.centroids:
                print centroid

            prev_dist, current_dist = current_dist, Xrdd.map(lambda (l, u): np.linalg.norm(u - self.centroids[l])).sum()
  
    def _update_centroids(self, X):
        sumByKey = X.combineByKey(self._combiner, self._merge_value, self._merge_combiners)
        averageByKey = sumByKey.map(self._mean).collect()
        return averageByKey

    def _combiner(self, u):
        """_combiner function for combineByKey to compute average centroid by cluster group"""
        return (u, 1)

    def _merge_value(self, u, value):
        """_merge_value function for combineByKey to compute average centroid by cluster group"""
        return (u[0] + value, u[1] + 1)

    def _merge_combiners(self, u, v):
        """_merge_combiners function for combineByKey to compute average centroid by cluster group"""
        return (u[0] + v[0], u[1] + v[1])

    def _mean(self, (l, (u, count))):
        """_compute mean value by key when rdd is in the form: (label, (cluster_sum, cluster_count))"""
        return u / count

    def _assign_data_to_centroids(self, u):
        """Assign data points to current centroids. To be applied to a Spark RDD foreach() method. 
        INPUT: u, row vector in X
        """
        return np.argmin( [np.linalg.norm(u - centroid) for centroid in self.centroids] )

    def _has_converged(self, old_labels, labels):
        remainder = (old_labels - labels).sum()
        if remainder > 0:
            return False
        else:
            return True


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

