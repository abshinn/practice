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

import matplotlib.pyplot as plt
import seaborn as sns
sns.axes_style("darkgrid")

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV

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
    os.system("cat DATA/adult/adult.data DATA/adult/adult.test > DATA/adult/adult.all")
    os.system("cat DATA/adult/adult.all | sed 's/, /,/g' > DATA/adult/adult.tmp")
    os.system("cat DATA/adult/adult.tmp | sed 's/K\./K/g' > DATA/adult/adult.csv")


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

    def __init__(self, estimator, binarize=True):
        self.data, self.label = self.prepare(binarize=binarize)
        self.estimator = estimator


    def prepare(self, binarize=True):

        names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                 "hours-per-week", "native-country", "label"]

        if os.path.isfile("DATA/adult/adult.csv") == False:
            download_data()

        datadf = pd.read_csv("DATA/adult/adult.csv", header = None, na_values = ['?'], names = names)

        del datadf["fnlwgt"]
        del datadf["workclass"]
        del datadf["native-country"]
        del datadf["race"]

        if binarize:
            data = binarize_df(datadf.dropna())
            label = data.pop(">50K")
            del data["<=50K"]
        else:
            data = datadf.dropna()
            label = data.pop("label")
            label = pd.Series(label == ">50K")

        return data, label


    def train(self, n_examples=None):

        X = self.data.values.astype(np.float32)

        y = self.label.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.estimator.fit(X_train, y_train)

        y_pred = self.estimator.predict(X_test)
        print classification_report(y_test, y_pred, target_names=["<=50k", ">50k"])

        y_score = self.estimator.predict_proba(X_test)
        print "roc: {}".format( roc_auc_score(y_test, y_score[:,1]) )


    def cv(self, parameters, scoring="roc_auc"):

        X = self.data.values.astype(np.float)
        y = self.label.values

        print cross_val_score(self.estimator, X, y, scoring=scoring, cv=3)


    def grid_search(self, param_grid, scoring=None, cv=3):

        X = self.data.values.astype(np.float)
        y = self.label.values

        grid_clf = GridSearchCV(self.estimator, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
        grid_clf.fit(X, y)

        print grid_clf.best_params_
        print grid_clf.best_score_

        self.estimator = grid_clf.best_estimator_


    def experience_curve(self, train_sizes=None, cv=3, ylim=None, scoring="roc_auc"):

        X = self.data.values.astype(np.float32)
        y = self.label.values

        if not train_sizes:
            train_sizes = np.linspace(.1, 1.0, 10)

        plt.figure()
        plt.title(">50K learning curve")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            self.estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring=scoring)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")

        return plt


    def importance(self):
        clf = ExtraTreesClassifier()
        y = self.label.values
        X = self.data.values
        clf.fit(X, y)
        for imp, col in sorted(zip(clf.feature_importances_, self.data.columns), key = lambda (imp, col): imp, reverse=True):
            print "[{:.5f}] {}".format(imp, col)



if __name__ == "__main__":
    estimator = DecisionTreeClassifier()
    fifty = FiftyK(estimator)

    parameters = {"max_features":(.1, .2, .3, "sqrt", "log2"),
                  "max_depth":(5, 8, 10, 12, 14, 16),
                  "min_samples_split":(2 ,3, 4),
                  "min_samples_leaf":(1, 2, 3),
                 }

    fifty.grid_search(param_grid=parameters, scoring=None, cv=5)

    fifty.train()
    fifty.experience_curve().show()

