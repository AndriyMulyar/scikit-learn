# cython: profile=True
import numpy as np
cimport numpy as np
"""
A meta file to hold wrapper classes
"""


class DecisionNode:
    """
    A class to encapsulate a Decision Node on a tree
    """
    def __init__(self, left_boundary_index, right_boundary_index, depth):
        """

        :param instances_in_node: The number of instances in the given DecisionNode
        :param left_index: The leftmost boundary in instances[] for the given node
        :param right_index: The rightmost boundary in instances[] for the given node
        """
        self.instances_in_node = 1+right_boundary_index-left_boundary_index
        self.left_boundary_index = left_boundary_index
        self.right_boundary_index = right_boundary_index

        self.left_child = None #left child of Decision Node
        self.right_child = None #right child of Decision Node
        self.is_leaf = False    #Whether a node is a leaf or not

        self.classification = None

        self.depth = depth


        self.split_record = None  #a split record object holding the best oblique split found on DecisionNode

class ObliqueSplitRecord:
    """
    Holds information about a bi-partition of the decision space
    """
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.hyperplane = np.zeros(dimensions+1, np.double) #coefficient hyperplane

        self.impurity_total = None  # stores the impurity of the current node
        self.impurity_left = None  #stores the impurity of left child
        self.impurity_right = None #stores the impurity of right child

        self.number_of_instances_left = None #stores the number of instances in left child
        self.number_of_instances_right = None  # stores the number of instances in right child

        self.criterion_used = None

        self.split_index = None     #index in instances[] that defines the bi-partition of instances at the given node
                                    #instances[split_index] is in left split, instances[split_index+1] in right split
                                    #notice node.left_boundary_index < split_index <= node.right_boundary_index

    def get_goodness_of_split(self):
        if self.impurity_left is None or self.impurity_right is None:
            return -1
        total = self.number_of_instances_left + self.number_of_instances_right
        return self.impurity_total - (self.number_of_instances_left*self.impurity_left + self.number_of_instances_right*self.impurity_right)/total

cpdef double hyperplane_compare(vector, hyperplane):
    """
    Compares the location of vector relative to hyperplane

    :param vector: a vector of n elements to get the relative position of
    :param hyperplane: the hyperlane (a vector with n+1 elements)
    :return: negative number if vector is below hyperplane, 0 if on hyperplane, positive number if above hyperplane
    """
    cdef double dot
    cdef int i
    dot = 0
    for i in range(len(vector)):
        dot += vector[i]*hyperplane[i]

    return dot+hyperplane[-1]