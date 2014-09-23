#!/usr/bin/env python2.7 -B -tt
"""
Implement a linked list
"""

import random

class Node(object):
    """Node of a linked list."""

    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next

    def __str__(self):
        return str(self.data)


class DoubleNode(object):
    """Node of a double linked list."""

    def __init__(self, data=None, prev=None, next=None):
        self.data = data
        self.prev = prev
        self.next = next

    def __str__(self):
        return str(self.data)


def print_list(node):
    """Given first node, print linked list."""

    while node:
        print node,
        node = node.next
    print


def remove_duplicates(unsorted_llist):
    """
    Cracking the Coding Inverview Problem 2.1, p77:
    
    Remove duplicates from an unsorted linked list.
    """
    pass

 
def generate_random_llist(n_elements=10, range=(0,10)):
    """Generate a random, unsorted linked list of integers."""

    llist = [ Node(random.randint(*range)) ]

    for i in xrange(n_elements-1):
        llist.append( Node(random.randint(*range)) )
        if i < n_elements-1:
            llist[i].next = llist[i+1]

    return llist


if __name__ == "__main__":
    node1, node2, node3 = Node(1), Node(2), Node(3)
    node1.next = node2
    node2.next = node3

    print_list(node1)


    llist = generate_random_llist()
    print_list(llist[0])
