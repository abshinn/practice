#!/usr/bin/env python2.7 -B -tt
"""
FizzBuzz

Write a program that prints the numbers from 1 to 100. But for multiples of three print 
"Fizz" instead of the number and for the multiples of five print "Buzz". For numbers 
which are multiples of both three and five print "FizzBuzz".
"""

def fizzbuzz(start = 1, end = 100):
    """fizzbuzz

    Print integers from start to end, but for multiples of 3, print Fizz, and for multiples
    of 5, print Buzz. For multiples of both 3 and 5, print FizzBuzz.

    INPUT: start, end
    OUTPUT: None -- print to stdout
    """

    for i in xrange(start, end+1):
        output = ""
        if i % 3 == 0: output += "Fizz"
        if i % 5 == 0: output += "Buzz"
        if output:
            print output,
        else:
            print i,


if __name__ == "__main__":
    fizzbuzz()
