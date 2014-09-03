#!/usr/bin/env python2.7 -B -tt
"""
Reservoir Sampling

Given a data stream of unknown size n, pick an entry uniformly at random so that each entry has a 1/n chance of being chosen.
"""

import numpy as np

def reservoir_sample(stream):

    for i, x in enumerate(stream, start=1):

        if np.random.rand() < 1.0 / i:
            keep = x

    return keep 


if __name__ == "__main__":
    stream = np.array([2, 3, 2, 2, 4, 5, 6, 2, 3, 1, 1, 2, 1, 2, 3, 3, 4, 3, 2, 1, 2, 6, 3, 3, 1, 0, 1, 0, 0, 2])

    sample = []
    for _ in xrange(300):
        sample.append(reservoir_sample(stream))

    print "true mean: {}\nsample mean: {}".format(stream.mean(), np.mean(sample))
