#!/usr/bin/env python2.7 -B -tt
""" Electrical Energy Output
http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant

    features: 4 (Temp, Ambient Pressure, Relative Humidity, Exhaust Vacuum), predict on Energy Output
    examples: 9568 over 6 years
feature type: float
        task: predict net hourly electrical energy output

"""

import os
import numpy as np
import pandas as pd
import pdb

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
# from sklearn.preprocessing import PolynomialFeatures

# model selection
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression


def download_data():
    """fetch data with wget"""

    baseurl = "http://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
    os.system("mkdir -p DATA/energy/")
    os.system("wget {} -P DATA/energy/".format(baseurl))

    # unzip    
    os.system("unzip DATA/energy/CCPP -d DATA/energy/")


class EnergyOutput(object):

    def __init__(self):
        self.cv_dfs = self.prep_data()

    def prep_data(self):

        filename = "DATA/energy/CCPP/Folds5x2_pp.xlsx"

        if not os.path.isfile(filename):
            download_data()

        xl_file = pd.ExcelFile(filename)
        dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
        return dfs


    def train(self, models = [Ridge(), LinearRegression()]):

        # run models and print score
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
    print __doc__
    energy = EnergyOutput()
    energy.train()
