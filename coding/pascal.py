#!/usr/bin/env python2.7 -B -tt
"""
Pascal's Triangle

Implement pascal's triangle. Given an integer index, return the corresponding
row of Pascal's Triangle.
"""

import sys


def pascal(index):
    """
    INPUT: integer index
    OUTPUT: list of integers, corresponding row of Pascal's Triangle
    """
    p = [1]
    for _ in xrange(index):
        p_a = p + [0]
        p_b = [0] + p
        p = []
        for i in xrange(len(p_a)):
            p.append(p_a[i] + p_b[i])
    return p


def test_pascal():
    ''' Test pascal function. '''
    print "pascal function...",
    assert pascal(0) == [1]
    assert pascal(5) == [1, 5, 10, 10, 5, 1]
    assert pascal(10) == [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]
    print "OK"


if __name__ == "__main__":
    test_pascal()

    # if desired, run as script
    # with index as kwarg
    if len(sys.argv) > 1:
        print pascal(int(sys.argv[1]))
    else:
        for i in xrange(10):
            print pascal(i)
