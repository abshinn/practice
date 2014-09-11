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


def fib_generator(index):
    """
    Generate the fibonacci squence up to index i.

    INPUT: integer
    OUTPUT: integer generator
    """
    current, previous = 1, 0
    i = 0
    while i < index: 
        yield previous
        current, previous = current + previous, current
        i += 1


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


def fib_even(x):
    """
    Generate even fibonacci numbers less than or equal to x.

    INPUT: integer
    OUTPUT: integer generator
    """
    current, previous = 1, 0
    while current < x:
        current, previous = current + previous, current
        if current % 2 == 0:
            yield current


if __name__ == "__main__":
    # test sequence functions
    print "        brute: {}".format( [fib_recursive(index) for index in xrange(13)] )
    print "    recursive: {}".format( [    fibonacci(index) for index in xrange(13)] )

    # test generator 
    print "fib_generator: {}".format( [x for x in fib_generator(13)] )
    print "         even: {}".format( [x for x in fib_even(144)] )
    print "sum even < 4M: {}".format( sum(fib_even(4000000)) ) 

