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
    end =   ">}])"

    input_list = list(input_string)
    for char in input_string:
        if char in start:
            endchar = end[start.index(char)]
            if endchar in input_list:
                input_list.remove(char)
                input_list.remove(endchar)
    if input_list:
        return False
    else:
        return True


if __name__ == "__main__":
    test = ["[]", "<{[]}>", "[]([])", "<[]<()", "[(])", "{}}"]
    for item in test:
        print closure(item)
