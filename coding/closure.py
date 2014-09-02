#!/usr/bin/env python2.7 -B -tt
"""
Closure

Write a function which takes a list of strings, return True if "<{[()]}>" are closed within
the string, False if otherwise.
"""

def closure(string):
    """
    INPUT: string
    OUTPUT: boolean
    """
    start = "<{[("
    end =   ">}])"

    char_list = list(string)
    for char in string:
        if char in start:
            endchar = end[start.index(char)]
            if endchar in char_list:
                char_list.remove(char)
                char_list.remove(endchar)
    if char_list:
        return False
    else:
        return True


if __name__ == "__main__":
    test = ["[]", "<{[]}>", "[]([])", "<[]<()", "[(])", "{}}"]
    for item in test:
        print closure(item)
