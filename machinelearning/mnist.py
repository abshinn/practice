#!/usr/bin/env python2.7 -B -tt
""" Employ a neural network to classify letters in the MNIST data set. """

def download_data():
    """ Download data with wget. """

    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    os.system("mkdir -p DATA/mnist/")
    os.system("wget {} -P DATA/mnist/".format(url))

if __name__ == "__main__":
    pass
