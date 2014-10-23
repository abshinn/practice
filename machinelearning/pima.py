#!/usr/bin/env python2.7 -B -tt
""" UCI Pima Indians Diabetes Classification

    http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

    features: 8
    examples: 768
feature type: numerical
        task: classification, clustering
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.axes_style("darkgrid")

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def download_data():
    """ Download data with wget. """

    baseurl = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes"
    os.system("mkdir -p DATA/pima/")
    os.system("wget {} -P DATA/pima/".format(baseurl + ".names"))
    os.system("wget {} -P DATA/pima/".format(baseurl + ".data"))


class Pima(object):
    """ UCI Pima Indians dataset. """

    def __init__(self, estimator):
        self.data, self.label = self._prepare()
        self.estimator = estimator

    def _prepare(self):
        """ Return tuple: (data, label). """

        filepath = "DATA/pima/pima-indians-diabetes.data"

        if os.path.isfile(filepath) == False:
            download_data()

        names = ["n_preg", "plasma", "blood_pressure", "triceps", "insulin", "bmi", "dpf", "age", "label"]

        data = pd.read_csv(filepath, header=None, names=names)

        data.blood_pressure = data.blood_pressure.apply(lambda x: x if x else np.nan)
        data.bmi = data.bmi.apply(lambda x: x if x else np.nan)
        data = data.dropna()

        label = data.pop("label")

        return data, label

    def train(self, n_examples=None):
        """ Train and fit data with self.estimator and show classification report. """

        X = self.data.values.astype(np.float32)

        y = self.label.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.estimator.fit(X_train, y_train)

        y_pred = self.estimator.predict(X_test)
        print classification_report(y_test, y_pred, target_names=["Not Diabetic", "Diabetic"])

        y_score = self.estimator.predict_proba(X_test)
        print "roc: {}".format( roc_auc_score(y_test, y_score[:,1]) )

    def grid_search(self, param_grid, scoring=None, cv=3):
        """ Run a grid-search with self.estimator through param_grid parameter space. """

        X = self.data.values.astype(np.float)
        y = self.label.values

        grid_clf = GridSearchCV(self.estimator, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
        grid_clf.fit(X, y)

        print grid_clf.best_params_
        print grid_clf.best_score_

        self.estimator = grid_clf.best_estimator_

    def experience_curve(self, train_sizes=None, cv=3, ylim=None, scoring="roc_auc"):
        """ Return matplotlib plt object with learning/experience curve using self.estimator. """

        X = self.data.values.astype(np.float32)
        y = self.label.values

        if not train_sizes:
            train_sizes = np.linspace(.1, 1.0, 10)

        plt.figure()
        plt.title("Pima")
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

    def plot_roc_curve(self):
        """ Return matplotlib plt object with a roc-auc curve. """

        X = self.data.values.astype(np.float32)
        y = self.label.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        probas_ = estimator.fit(X_train, y_train).predict_proba(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, probas_[:,1])
        roc_auc = auc(fpr, tpr)
        print "roc: {}".format(roc_auc)

        plt.plot(fpr, tpr, label="ROC curve (area = {})".format(roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Pima Indians Diabetes Classification")
        plt.legend(loc="lower right")
        plt.axis("equal")

        return plt

    def reduce_dimension(self, n_components=2):
        """ Return PCA transform of self.data, with n_components. """

        reducer = PCA(n_components=n_components)

        X = self.data.values

        norm = Normalizer()
        Xnorm = norm.fit_transform(X)

        return reducer.fit_transform(Xnorm)

    def plot(self):
        """ Return matplotlib plt object of scatter plot using first two features. """

        X = self.reduce_dimension(n_components=None)

        plt.figure()
        plt.scatter(X[:,0], X[:,1])

        return plt

    def cluster(self):
        """ Cluster data with reduced dimensions. """

        X = self.reduce_dimension(n_components=2)
    
        cv = cross_val_score(self.estimator, X, cv=5)
        print cv
        print "mean: {}\nstd: {}".format(np.mean(cv), np.std(cv))

def forest_search():
    """ Random Forest Grid Search. """

    estimator = RandomForestClassifier()
    parameters = {"n_estimators":(10, 20, 30, 40), 
                  "max_features":(.1, .2, .3, "sqrt", "log2"),
                  "max_depth":(5, 8, 10, 12, 14, 16),
                  "min_samples_split":(2 ,3, 4),
                  "min_samples_leaf":(1, 2, 3),
                 }

    pima.grid_search(param_grid=parameters, scoring=None, cv=5)
    pima.train()
    pima.experience_curve().show()
    pima.plot_roc_curve().show()



if __name__ == "__main__":
    seed = np.random.randint(2**32)
    print "seed: {}".format(seed)

    estimator = RandomForestClassifier(max_features=0.3, min_samples_split=2, n_estimators=32, max_depth=3, min_samples_leaf=1, random_state=seed)

    pima = Pima(estimator)
    pima.train()
    pima.experience_curve().show()
    pima.plot_roc_curve().show()

