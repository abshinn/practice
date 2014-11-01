#!/usr/bin/env python2.7 -B -tt
""" Employ a neural network to classify letters in the MNIST data set. """

import cPickle
import gzip
import numpy as np

def download_data():
    """ Download data with wget. """

    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    os.system("mkdir -p DATA/mnist/")
    os.system("wget {} -P DATA/mnist/".format(url))

def load_mnist():
    """ Load and unpickle mnist data set. """

    with gzip.open("DATA/mnist/mnist.pkl.gz", "rb") as f:
        train_set, valid_set, test_set = cPickle.load(f)

if __name__ == "__main__":
    load_mnist()

