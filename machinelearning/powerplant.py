#!/usr/bin/env python2.7 -B -tt
""" Combined Cycle Power Plant Data Set

    http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant

    features: 4 (Temp, Ambient Pressure, Relative Humidity, Exhaust Vacuum), predict on Energy Output
    examples: 9568 over 6 years
feature type: float
        task: predict net hourly electrical energy output
"""

import os
import numpy as np
import pandas as pd

from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
sns.axes_style("darkgrid")

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
# from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def download_data():
    """ Fetch data with wget and unzip. """

    baseurl = "http://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
    os.system("mkdir -p DATA/energy/")
    os.system("wget {} -P DATA/energy/".format(baseurl))
    os.system("unzip DATA/energy/CCPP -d DATA/energy/")


class EnergyOutput(object):
    """ """

    def __init__(self, regressor):
        self.X, self.y = self._prepare()
        self.regressor = regressor 

    def _prepare(self):

        filename = "DATA/energy/CCPP/Folds5x2_pp.xlsx"

        if not os.path.isfile(filename):
            download_data()

        exceldf = pd.ExcelFile(filename)
#         dfdict = {sheet_name: exceldf.parse(sheet) for sheet in exceldf.sheet_names}
        data = pd.concat([exceldf.parse(sheet) for sheet in exceldf.sheet_names]).values
        X = data[:,:-1].astype(np.float32)
        y = data[:,-1]

        return (X, y)

    def experience_curve(self, train_sizes=None, cv=5, ylim=None, scoring="mean_squared_error"):
        """ Return matplotlib plt object with learning/experience curve using self.estimator. """

#         X = normalize(self.X)

        if not train_sizes:
            train_sizes = np.linspace(.1, 1.0, 10)

        plt.figure()
        plt.title("UCI Energy Output")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            self.regressor, self.X, self.y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring=scoring)

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

    def train(self, models = [Ridge(), LinearRegression()]):

        for model in models:
            scores = []
            for cvset in self.cv_dfs.values():
                data = cvset.values
                X = data[:,:-1]
#                 X = normalize(X)
                y = data[:,-1]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 42)

                model.fit(X_train, y_train)
                scores.append( model.score(X_test, y_test) )

            avg_score = np.mean(scores)

            print model
            print "\t==> Rsqr Score: {}\n".format(avg_score)


if __name__ == "__main__":
    energy = EnergyOutput(DecisionTreeRegressor(max_depth=4))
    energy.experience_curve().show()

