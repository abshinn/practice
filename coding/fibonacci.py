#!/usr/bin/env python2.7 -B -tt
"""
fibonacci coding problems

sequence:
0, 1, 1, 2, 3, 5, 8, 13, 21, 34...
"""

def fib(index):
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

def test_fib():
    assert fib(0) == 0
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(8) == 21
    assert fib(10) == 55
    print "fib OK"


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

def test_fib_generator():
    fibs = [fib for fib in fib_generator(10)]
    assert fibs[0] == 0
    assert fibs[3] == 2
    assert fibs[-1] == 34
    print "fib_generator OK"


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

def test_fib_recursive():
    assert fib_recursive(0) == 0
    assert fib_recursive(1) == 1
    assert fib_recursive(2) == 1
    assert fib_recursive(8) == 21
    assert fib_recursive(10) == 55
    print "fib_recursive OK"


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


def test_fib_even():
    even_fibs = [even_fib for even_fib in fib_even(144)]
    assert even_fibs[0] == 2
    assert even_fibs[3] == 144
    print "fib_even OK"


if __name__ == "__main__":
    test_fib()
    test_fib_recursive()
    test_fib_generator()
    test_fib_even()


    # sequence functions
    print "        brute: {}".format( [fib_recursive(index) for index in xrange(13)] )
    print "    recursive: {}".format( [          fib(index) for index in xrange(13)] )

    # generators 
    print "fib_generator: {}".format( [x for x in fib_generator(13)] )
    print "         even: {}".format( [x for x in fib_even(144)] )
    print "sum even < 4M: {}".format( sum(fib_even(4000000)) ) 

