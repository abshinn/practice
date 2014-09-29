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

from sklearn.cluster import KMeans


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

    def prepare(self):
        """prepare data set for trianing, assign data to instance variable"""

        if os.path.isfile("DATA/diabetes/admission_type_id.csv") == False:
            download_data()

        # load data
        id_mapping = pd.read_csv("DATA/diabetes/admission_type_id.csv", index_col = 0)
        data = pd.read_csv("DATA/diabetes/diabetic_data.csv")

        # binarize admission type
        admdf = pd.DataFrame()
        for adtype, ad_id in zip(id_mapping.description, id_mapping.index):
            admdf[adtype] = (data.admission_type_id == ad_id)

        # binarize categorical text columns
        catdf = pd.DataFrame()
        dtype = data.race.dtype # grab datatype
        features = ["race", "gender", "weight", "age", "diabetesMed", "insulin", "change", "readmitted"]
        for feature in features:
            if data[feature].dtype == dtype:
                catdf = pd.concat([catdf, binarize(data[feature])], axis = 1)
            else:
                catdf = pd.concat([catdf, data[feature]], axis = 1)

        # choose non-binary columns
        nonbindf = data[["num_medications", "num_procedures", "num_lab_procedures", "number_outpatient", 
                         "number_emergency", "number_inpatient", "number_diagnoses"]]

        self.data = data
        self.traindf = pd.concat([catdf, admdf, nonbindf], axis = 1)

    def train(self):
        """train data using a sklearn clustering algorithm"""

        self.prepare()

        inertias = []

        for N in xrange(2, 9):
            km = KMeans(n_clusters = N)
            km.fit(self.traindf.values)
            inertias.append(km.inertia_)

        print inertias
        plt.plot(inertias, maker="o")
        plt.show()

        print "==> Running Kmeans on data set of shape: {}".format(self.traindf.shape)
        km = KMeans(n_clusters = 3)
        km.fit(self.traindf.values)
        self.klabels = km.labels_
        self.inertia = km.inertia_


    def print_clusters(self):
        """print clusters"""

        for k in xrange(self.n_clusters):
            print "\n      ======== cluster {} / {} ========".format(k, (self.klabels == k).sum())
            for column in self.traindf.columns:
                if column not in bools:
                    continue
                if self.traindf[column].dtype == np.dtype("bool"):
                    percent = self.traindf.loc[self.klabels == k, column].sum()/np.float(self.traindf[column].sum())
                    print "{:>20}: {:.3f}".format(column, percent)


if __name__ == "__main__":
    print __doc__
    pc = PatientCluster(n_clusters = 4)
    pc.train()

