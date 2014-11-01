#!/usr/bin/env python2.7 -B -tt
""" Employ deep learning to classify letters in the MNIST data set. """

import cPickle
import gzip
import theano
import theano.tensor as T
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

    return train_set, valid_set, test_set

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as "floatX" as well
    # ("shared_y" does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # "shared_y" we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, "int32")

def main():
    train_set, valid_set, test_set = load_mnist()

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    batch_size = 500    # size of the minibatch

    # accessing the third minibatch of the training set

    data  = train_set_x[2 * 500: 3 * 500]
    label = train_set_y[2 * 500: 3 * 500]


if __name__ == "__main__":
    main()

