#!/usr/bin/env python2.7 -B -tt
""" Linked List Practice. """

import random

        
class Node(object):
    """Node of a linked list."""

    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next

    def __str__(self):
        return str(self.data)


class DoubleNode(Node):
    """Node of a double linked list."""

    def __init__(self, data=None, prev=None, next=None):
        self.data = data
        self.prev = prev
        self.next = next


def print_list(node):
    """Given first node, print linked list."""

    while node:
        print node,
        node = node.next
    print


class LinkedList(object):
    """Linked List"""

    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.head = nodes[0]

    def show(self):
        print_list(self.head)


def split_llist(llist):
    """Divide a linked list into two."""
    
    marker1, marker2 = 0, 0

    node = llist.head

    while node:
        marker1 += 1
        marker2 = marker1/2
        node = node.next

    A_nodes = []
    node = llist.head
    for _ in xrange(marker2):
        A_nodes.append(node)
        node = node.next

    A_nodes[-1].next = None

    B_nodes = []
    for _ in xrange(marker2,marker1):
        B_nodes.append(node)
        node = node.next
       
    return LinkedList(A_nodes), LinkedList(B_nodes)


def sort_llist(unsorted_llist):
    """Sort an unsorted linked list."""
    pass


def remove_duplicates(unsorted_llist):
    """
    Cracking the Coding Inverview Problem 2.1, p77:
    
    Remove duplicates from an unsorted linked list.
    """
    pass

 
def generate_random_llist(n_elements=10, range=(0,10)):
    """Generate a random, unsorted linked list of integers.
    
    INPUT: n_elements -- number of elements in linked list (integer)
                range -- range of random integers to be node values (tuple of two integers)
    OUTPUT: tuple: (head Node of linked list, list of nodes)
    """

    llist = [ Node(random.randint(*range)) ]

    for i in xrange(n_elements-1):
        llist.append( Node(random.randint(*range)) )
        if i < n_elements-1:
            llist[i].next = llist[i+1]

    return LinkedList(llist)


if __name__ == "__main__":
    node1, node2, node3 = Node(1), Node(2), Node(3)
    node1.next = node2
    node2.next = node3
    print_list(node1)

    llist = generate_random_llist(range=(1,5))
    print "random list:",
    llist.show()
  
    llistA, llistB = split_llist(llist)

    print "A:",
    llistA.show()

    print "B:",
    llistB.show()
