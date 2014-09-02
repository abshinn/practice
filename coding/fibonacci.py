#!/usr/bin/env python2.7 -B -tt
"""
Fibonacci
0, 1, 1, 2, 3, 5, 8, 13, 21...

Write a function which takes an integer index and returns the corresponding integer in the fibonacci sequence.
"""

def fibonacci(index):
    if index <= 1:
        return index
    previous = 0
    current = 1
    for _ in xrange(index - 1):
        current, previous = current + previous, current
    return current


def fib_recursive(index):
    if index <= 1:
        return index
    previous = fib_recursive(index - 2)
    current = fib_recursive(index - 1)
    return current + previous


if __name__ == "__main__":
    for x in xrange(12):
        print fib_recursive(x)
