#!/usr/bin/env python2.7 -B -tt
"""
fibonacci coding problems

sequence:
0, 1, 1, 2, 3, 5, 8, 13, 21, 34...
"""

def fibonacci(index):
    """
    Write a function which takes an integer index and returns the corresponding integer in the fibonacci sequence.

    INPUT: integer
    OUTPUT: integer
    """
    if index < 2:
        return index
    current, previous = 1, 0
    for _ in xrange(index - 1):
        current, previous = current + previous, current
    return current


def fib_recursive(index):
    """
    Write a recursive function which takes an integer index and returns the corresponding integer in the fibonacci sequence.

    INPUT: integer
    OUTPUT: integer
    """
    if index < 2:
        return index
    previous = fib_recursive(index - 2)
    current = fib_recursive(index - 1)
    return current + previous


def fib_even(n):
    """
    Generate even fibonacci numbers less than n.

    INPUT: integer
    OUTPUT: integer generator
    """
    current, previous = 1, 0
    while current < n:
        current, previous = current + previous, current
        if current % 2 == 0:
            yield current


if __name__ == "__main__":
    # test sequence functions
    for fib in [fibonacci, fib_recursive]:
        fib_list = []
        for x in xrange(13):
            fib_list.append( fib(x) )
        print fib_list

    # test generator 
    for x in fib_even(10):
        print x
    print sum( fib_even(4000000) )

