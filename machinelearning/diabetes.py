#!/usr/bin/env python2.7 -B -tt
""" Patient Clustering: UCI Diabetes Dataset
https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

    features: 55
    examples: 100,000
feature type: mixed categorical
        task: cluster patient types and categorize the clusters
"""

import os
import numpy as np
import pandas as pd
import pdb

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier


def download_data():
    os.system("wget http://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip -P DATA/")
    os.system("tar xvf DATA/dataset_diabetes.zip")
    os.system("mv dataset_diabetes.zip DATA/diabetes/")
    os.system("head -9 DATA/diabetes/IDs_mapping.csv > DATA/diabetes/admission_type_id.csv")


def binarize(series):
    """binarize a pandas series of categorical strings into a sparse dataframe"""
    name = series.name
    df = pd.DataFrame()
    for category in series.value_counts().index:
        df[category] = (series == category)
    return df


bools = [u'Caucasian', u'AfricanAmerican', u'Hispanic', u'Other', u'Asian', u'Female', u'Male', u'Unknown/Invalid', 
         u'[75-100)', u'[50-75)', u'[100-125)', u'[125-150)', u'[25-50)', u'[0-25)', u'[150-175)', u'[175-200)', u'>200', 
         u'[70-80)', u'[60-70)', u'[50-60)', u'[80-90)', u'[40-50)', u'[30-40)', u'[90-100)', u'[20-30)', u'[10-20)', u'[0-10)',
         u'Emergency', u'Urgent', u'Elective', u'Newborn', u'Trauma Center'] 


class PatientCluster(object):
    """patient clustering class"""

    def __init__(self, n_clusters = 5):
        self.n_clusters = n_clusters
        self._prepare()

    def _prepare(self):
        """prepare data set for trianing, assign data to instance variable"""

        if os.path.isfile("DATA/diabetes/admission_type_id.csv") == False:
            download_data()

        id_mapping = pd.read_csv("DATA/diabetes/admission_type_id.csv", index_col = 0)
        data = pd.read_csv("DATA/diabetes/diabetic_data.csv")

        # binarize admission type
        admdf = pd.DataFrame()
        for adtype, ad_id in zip(id_mapping.description, id_mapping.index):
            admdf[adtype] = (data.admission_type_id == ad_id)

        # binarize categorical text columns
        catdf = pd.DataFrame()
        dtype = data.race.dtype # grab datatype
        features = ["race", "gender", "age", "diabetesMed", "insulin", "change", "readmitted"]
        for feature in features:
            if data[feature].dtype == dtype:
                catdf = pd.concat([catdf, binarize(data[feature])], axis = 1)
            else:
                catdf = pd.concat([catdf, data[feature]], axis = 1)

        # choose non-binary columns
        nonbindf = data[["num_medications", "num_procedures", "num_lab_procedures", "number_outpatient", 
                         "number_emergency", "number_inpatient", "number_diagnoses"]]

        self.data = pd.concat([catdf, admdf, nonbindf], axis = 1)

    def elbow(self, nrange=(2,9)):
        """train data on multiple cluster sizes and return elbow plot"""

        inertias = []
        for N in xrange(*nrange):
            km = KMeans(n_clusters = N)
            km.fit(self.data.values)
            inertias.append(km.inertia_)

        print inertias
        plt.plot(range(*nrange), inertias, marker="o", markerfacecolor="purple")
        plt.title("cluster elbow")
        plt.ylabel("KMeans inertia")
        plt.xlabel("n_clusters")

        return plt

    def reduce_dimension(self, n_components=2):
        """use principal component analysis to reduce features down to a viewable dimension"""

        reducer = PCA(n_components=n_components)

        X = self.data.values.astype(np.float32)

        norm = Normalizer()
        Xnorm = norm.fit_transform(X)

        return reducer.fit_transform(Xnorm)

    def scatter_plot(self):
        """scatter plot of data with reduced dimensions"""

        X = self.reduce_dimension(n_components=2)

        plt.figure()
        plt.scatter(X[:,0], X[:,1])

        return plt

    def train(self):
        """train data using a sklearn clustering algorithm"""

        print "==> Running Kmeans on data set of shape: {}".format(self.data.shape)
        km = KMeans(n_clusters = self.n_clusters)
        km.fit(self.data.values)
        self.labels = km.labels_
        self.inertia = km.inertia_

    def cluster_importance(self, clf=DecisionTreeClassifier(), n_most_important=3):
        """once trained, figure out the most important features for each cluster using a tree classifier"""

        for k in xrange(self.n_clusters):
            labels = (self.labels == k)
            clf.fit(self.data.values, labels)

            print "\n      ======== cluster {} / {} ========".format(k + 1, self.n_clusters)

            sorted_importance = sorted(zip(clf.feature_importances_, self.data.columns), key=lambda (imp, col): imp, reverse=True)
            sorted_importance = sorted_importance[:n_most_important]

            for imp, col in sorted_importance:
                print "[{:.5f} relative importance] {}".format(imp, col)
                print self.data.loc[labels, col].describe()

    def print_clusters(self):
        """print clusters"""

        for k in xrange(self.n_clusters):
            print "\n      ======== cluster {} / {} ========".format(k + 1, self.n_clusters)
            for column in self.data.columns:
                if column not in bools:
                    continue
                if self.data[column].dtype == np.dtype("bool"):
                    percent = self.data.loc[self.labels == k, column].sum()/np.float(self.data[column].sum())
                    print "{:>20}: {:.3f}".format(column, percent)


if __name__ == "__main__":
    print __doc__
    pc = PatientCluster(n_clusters = 3)
    pc.train()
    pc.cluster_importance()
