#!/usr/bin/env python2.7 -B -tt
"""
Implement sorting algorithms.
"""

from heapq import merge

def merge_sort(input):
    length = len(input)

    if length <= 1:
        return input

    middle = length / 2
    left  = input[:middle]
    right = input[middle:]

    left  = merge_sort(left)
    right = merge_sort(right)

    return list(merge(left, right))


if __name__ == "__main__":
    input = [10, 9, 8, 3, 4, 2, 6, 7, 1, 5]
    
    print " input: {}".format(input)
    print "output: {}".format(merge_sort(input))
