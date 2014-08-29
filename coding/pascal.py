#!/usr/bin/env python2.7 -B -tt
# Pascal's Triangle
# 
# Implement pascal's triangle. Given an integer index, return the corresponding row of Pascal's Triangle.

def pascal(ind):
    """
    INPUT: integer index
    OUTPUT: list of integers corresponding to a row of Pascal's Triangle
    """
    p = [1]
    for _ in xrange(ind):
        p_a = p + [0]
        p_b = [0] + p
        p = []
        for i in xrange(len(p_a)):
           p.append(p_a[i] + p_b[i])
    return p

if __name__ == "__main__":
    ind = int(raw_input())
    print pascal_brute(ind)
