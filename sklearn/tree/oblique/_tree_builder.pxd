import numpy as np
cimport numpy as np
from _splitter cimport ObliqueSplitRecord
"""
A Decision Node holds all information needed to decide the decision path that an un-seen instance should take

"""
cdef struct DecisionNode:
    np.int32_t left_child_node      #id of the left child node
    np.int32_t right_child_node     #id of the right child node
    np.int32_t number_of_instances  #number of instances at given node
    np.float32_t impurity           #total impurity of node

    ObliqueSplitRecord split
