#!/usr/bin/env python2.7 -B -tt
""" UCI Adult Dataset (1994 Census)
    https://archive.ics.uci.edu/ml/datasets/Adult

        features: 14
        examples: 48842
    feature type: categorical and numerical
            task: predict wheter a person makes above 50k a year
"""

import os
import numpy as np
import pandas as pd
import pdb

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def download_data():
    """fetch data with wget"""
    baseurl = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult"
    os.system("mkdir -p DATA/adult/")
    os.system("wget {} -P DATA/adult/".format(baseurl + ".names"))
    os.system("wget {} -P DATA/adult/".format(baseurl + ".data"))
    os.system("wget {} -P DATA/adult/".format(baseurl + ".test"))

    # remove spaces after comma for pandas convenience
    os.system("cat DATA/adult/adult.data | sed 's/, /,/g' > DATA/adult/adult.csv")
    os.system("cat DATA/adult/adult.test | sed 's/, /,/g' > DATA/adult/test.csv")


def binarize_df(dframe):
    """binarize a pandas series of categorical strings into a sparse dataframe"""
    dfout = pd.DataFrame()
    for column in dframe.columns:
        col_dtype = dframe[column].dtype
        if col_dtype == object:
            # assume categorical string
            for category in dframe[column].value_counts().index:
                dfout[category] = (dframe[column] == category)
        elif col_dtype == np.int or col_dtype == np.float:
            dfout[column] = dframe[column]
        else:
            print "unused column: {}".format(column)
    return dfout


class FiftyK(object):

    def __init__(self, estimator):
        self.data, self.label = self.data_prep()
        self.estimator = estimator


    def data_prep(self):
        names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                 "hours-per-week", "native-country", "label"]

        if os.path.isfile("DATA/adult/adult.csv") == False:
            download_data()

        datadf = pd.read_csv("DATA/adult/adult.csv", header = None, na_values = ['?'], names = names)

        data = binarize_df(datadf.dropna())
        label = data[">50K"]
        del data[">50K"]
        del data["<=50K"]
#         del data["fnlwgt"]
        return data, label


    def train(self):
        X = self.data.values.astype(np.float)
        y = self.label.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.estimator.fit(X_train, y_train)

        y_pred = self.estimator.predict(X_test)
        print classification_report(y_test, y_pred, target_names=["<50k", ">50k"])

        y_score = self.estimator.predict_proba(X_test)
        print "roc: {}".format( roc_auc_score(y_test, y_score[:,1]) )


    def cv(self):
        X = self.data.values
        y = self.label.values

        print cross_val_score(self.estimator, X, y, scoring="roc_auc", cv=3)


    def importance(self):
        clf = ExtraTreesClassifier()
        y = self.label.values
        X = self.data.values
        clf.fit(X, y)
        for imp, col in sorted(zip(clf.feature_importances_, self.data.columns), key = lambda (imp, col): imp, reverse = True):
            print "[{:.5f}] {}".format(imp, col)



if __name__ == "__main__":
    estimator = RandomForestClassifier(n_estimators=50)
    fifty = FiftyK(estimator)
    fifty.data_prep()
#     fifty.importance()
    fifty.train()
#     fifty.cv()
