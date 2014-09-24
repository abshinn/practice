#!/usr/bin/env python2.7 -B -tt
""" Linked list practice problems. """

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


class LinkedList(object):
    """Linked List."""

    def __init__(self):
        self.head = None

    def length(self):
        node = self.head
        l = 0
        while node:
            l += 1
            node = node.next
        return l

    def add(self, node_value):
        if self.head:
            node = self.head
            while node.next:
                node = node.next
            node.next = Node(node_value)
        else:
            self.head = Node(node_value)

    def split(self):
        """Divide a linked list into two linked lists."""
        
        marker_1, marker_2 = 0, 0

        node = self.head
        while node:
            marker_1 += 1
            marker_2 = marker_1/2
            node = node.next

        node = self.head
        list_A = LinkedList()
        for _ in xrange(marker_2):
            list_A.add(node.data)
            node = node.next

        list_B = LinkedList()
        for _ in xrange(marker_2, marker_1):
            list_B.add(node.data)
            node = node.next
           
        return list_A, list_B

    def remove(self, item):

        node = self.head
        prev = None
        while node:
            if node.data == item:
                if prev == None:
                    self.head = self.head.next
                else:
                    prev.next = node.next
                    prev, node = prev, node.next
            else:
                prev, node = node, node.next


    def _swap(self, node):
        """Swap the position of the given node with the next."""
        pass

    def show(self):
        node = self.head
        while node:
            print node,
            node = node.next
        print


def remove_duplicates(unsorted_llist):
    """
    Cracking the Coding Inverview Problem 2.1, p77:
    
    Remove duplicates from an unsorted linked list.
    (using temporary buffer)
    """
    pass

 
def generate_random_llist(n_elements=10, range=(0,10)):
    """Generate a random, unsorted linked list of integers.
    
    INPUT: n_elements -- number of elements in linked list (integer)
                range -- range of random integers to be node values (tuple of two integers)
    OUTPUT: tuple: (head Node of linked list, list of nodes)
    """

    llist = LinkedList()

    for _ in xrange(n_elements):
        llist.add( random.randint(*range) )

    return llist


if __name__ == "__main__":
    test = LinkedList()
    test.add(1)
    test.add(2)
    test.add(3)
    test.show()

    llist = generate_random_llist(n_elements=11, range=(1,5))
    print "random list:",
    llist.show()

    print "remove 1:",
    llist.remove(1)
    llist.show()

    print "\nSplit"
    llistA, llistB = llist.split()

    print "A:",
    llistA.show()

    print "B:",
    llistB.show()
