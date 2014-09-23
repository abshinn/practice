#!/usr/bin/env python2.7 -B -tt
"""
Implement a linked list
"""

class Node(object):
    """Node of a linked list"""

    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next

    def __str__(self):
        return str(self.data)


class doubleNode(object):
    """Node of a double linked list"""

    def __init__(self, data=None, prev=None, next=None):
        self.data = data
        self.prev = prev
        self.next = next

    def __str__(self):
        return str(self.data)


def print_list(node):
    """given first node, print linked list"""

    while node:
        print node,
        node = node.next
    print


if __name__ == "__main__":
    node1, node2, node3 = Node(1), Node(2), Node(3)
    node1.next = node2
    node2.next = node3

    print_list(node1)
