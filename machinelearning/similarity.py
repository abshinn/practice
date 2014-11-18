""" Module for calculating and visualizing similarity. """

import numpy as np
import matplotlib.pyplot as plt


def cosine_distance(data):
    """ Compute cosine distance similarity matrix using matrix multiplication.

    INPUT
        data -- numpy ndarray of shape (n, m)

    OUTPUT
        cos  -- numpy ndarray of shape (n, n)
    """

    data = (data - data.mean(axis=0))/data.std(axis=0)
    data[np.isnan(data)] = 0.

    similar = np.dot(data, data.T)
    
    square_mag = np.diag(similar)
    
    inv_square_mag = 1. / square_mag
    inv_square_mag[np.isinf(inv_square_mag)] = 0.

    inv_mag = np.sqrt(inv_square_mag)
    
    cosine = similar * inv_mag
    cosine = cosine.T * inv_mag

    return cosine


class simMatrix(object):
    """ Compute and display similarity matrix. """

    def __init__(self, ndarray, method=cosine_distance):
        self.data = method(ndarray)

    def plot(self):
        """ Return matplotlib plt object of similarity matrix. """

        plt.matshow(self.data, cmap="Reds")
        plt.colorbar()

        return plt

