#!/usr/bin/env python2.7 -B -tt
"""
Closure

Write a function which takes a list of strings, return True if "<{[()]}>" are closed within
the string, False if otherwise.
"""

def closure(input_string):
    """
    INPUT: string
    OUTPUT: boolean
    """
    start = "<{[("
    end = ")]}>"

    open_items = []
    input_list = list(input_string)
    for char in input_string:
        if char in start:
            open_items.append(char)
        if char in end and char in open_items:    
            input_list.remove(char)
            open_items.remove(char)

    


if __name__ == "__main__":
    closure([])
