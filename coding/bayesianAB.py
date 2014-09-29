#!/usr/bin/env python2.7 -B -tt
"""
Implement Bayesian AB Test.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette="Set2")
sns.set_context("poster")

beta_dist = np.random.beta


def distribution_plot():
    """
    Simulate a control and treatments, and plot their distributions.
    """

    # control
    Cconversions = 200
    Cviews = 1000

    # treatments
    T1conversions = 210
    T1views = 1000
    T2conversions = 150
    T2views = 1000

    n_samples = 100000 # number of samples to use in the distribution
    control    = beta_dist( Cconversions + 1,  Cviews -  Cconversions + 1, n_samples)
    treatment1 = beta_dist(T1conversions + 1, T1views - T1conversions + 1, n_samples)
    treatment2 = beta_dist(T2conversions + 1, T2views - T2conversions + 1, n_samples)

    proba1 = np.mean(treatment1 > control)
    proba2 = np.mean(treatment2 > control)

    # plot beta distributions
    sns.kdeplot(   control, shade=True, label=   "control")
    sns.kdeplot(treatment1, shade=True, label="treatment1")
    sns.kdeplot(treatment2, shade=True, label="treatment2")
    plt.legend()
    plt.title(".")

    print "Probability1: {}".format(proba1*100)
    print "Probability2: {}".format(proba2*100)

    return plt


def test_over_time():
    """
    Set up initial conditions, then run AB test over time.
    """

    Cviews, Tviews = 1000, 1000
    Cconversions, Tconversions = 200, 200
    n_samples = 100000 # number of samples to use in the distribution

    conversions = []
    probas = []
    views = []
    for _ in xrange(5000):
        Cviews += (20*np.random.rand())
        Tviews += (20*np.random.rand())
        views.append( (Cviews, Tviews) )

        Cconversions += (10*np.random.rand())
        Tconversions += (10*np.random.rand())
        conversions.append( (Cconversions, Tconversions) )
        
        treatment = beta_dist(Tconversions + 1, Tviews - Tconversions + 1, n_samples)
        control   = beta_dist(Cconversions + 1, Cviews - Cconversions + 1, n_samples)
        
        probas.append(np.mean(treatment > control))

    views = np.array(views)
    conversions = np.array(conversions)
        
    plt.plot(probas, marker="o", markerfacecolor="purple")
    plt.ylim((0,1))
    plt.title("Pr( treatement > control )")

    plt.figure()
    plt.plot(views[:,0]/conversions[:,0], color="red", label="control")
    plt.plot(views[:,1]/conversions[:,1], color="steelblue", label="treatment")
    plt.title("conversion rate")
    plt.legend();

    del views, conversions, probas

    return plt 



if __name__ == "__main__":
    distribution_plot().show()
    test_over_time().show()

