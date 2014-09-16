#!/usr/bin/env python2.7 -B -tt
"""
Closure

Write a function which takes a list of strings, return True if "<{[()]}>" are closed within the string,
False if otherwise.
"""

def simple_closure(string):
    """
    Detect if enclosure items have their compliment present. Return True if closed, False if otherwise.
    This method would return True for linked items, e.g., "([)]".

    INPUT: str
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


def code_closure(in_string):
    """
    Detect if enclosure items within a given string are closed, coding rules apply, e.g., "([)]" would
    return False.

    INPUT: str
    OUTPUT: boolean
    """
    if in_string == "":
        return True

    start = "<{[("
    end =   ">}])"

    i = 0
    while i < len(in_string): 
        if in_string[i] in start:
            endchar = end[start.index(in_string[i])]

            endchar_pos = in_string[i:].find(endchar)

            if endchar_pos == -1:
                return False
            else:
                if code_closure(in_string[i+1:endchar_pos]) and code_closure(in_string[endchar_pos+1:]):
                    return True
                else:
                    return False
        else:
            i += 1

def test_code_closure():
    assert code_closure("[{}]<>") == True
    assert code_closure("{}()<[]>") == True
    assert code_closure("<>{(})<[]>") == False
    assert code_closure("<{<[]>}") == False
    assert code_closure("((())))") == False
    print "clode_closure OK"


if __name__ == "__main__":
    test_code_closure()

    test = ["[]", "<{[]}>", "[]([])", "<[]<()", "[(])", "{}}"]
    for item in test:
        print code_closure(item)
