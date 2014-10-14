#!/usr/bin/env spark-submit
""" Implement KMeans using Lloyd's algorithm in Spark. """

from pyspark import SparkContext, SparkConf
import numpy as np
np.random.seed(42)

class KMeans(object):

    def __init__(self, k_clusters):
        self.k_clusters = k_clusters

    def fit(self, Xrdd):
        """ Fit data to k clusters using Lloyd's algorithm. """

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
        """ Update centroids by using the combineByKey RDD method. """
        sumByKey = X.combineByKey(lambda value: (value, 1),
                                  lambda x, value: (x[0] + value, x[1] + 1),
                                  lambda x, y: (x[0] + y[0], x[1] + y[1]))
        averageByKey = sumByKey.map(lambda (key, (value_sum, count)): value_sum / count).collect()
        return averageByKey

    def _assign_data_to_centroids(self, u):
        """ Assign data points to current centroids, input func to foreach() RDD method. 
        INPUT: u, row vector in X
        """
        return np.argmin( [np.linalg.norm(u - centroid) for centroid in self.centroids] )


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

