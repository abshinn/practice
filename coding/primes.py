#!/usr/bin/env python2.7 -B -tt
"""
Coding problems involving prime numbers.
"""

def prime_generator(x):
    """
    Generate all primes less than or equal to x.
    """
    prime = 2
    while prime <= x:
        test = True
        for mod in xrange(2, prime/2 + 1):
            if prime % mod == 0:
                test = False
                break 
        if test:
            yield prime

        prime += 1


if __name__ == "__main__":
    print [prime for prime in prime_generator(31)]
