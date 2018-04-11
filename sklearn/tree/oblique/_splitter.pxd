import numpy as np
cimport numpy as np

"""
A record of a generated Oblique split of a DecisionNode
The best ObliqueSplitRecord, the one with minimum (left_impurity + right_impurity), 
will become the ObliqueSplitRecord used in the DecisionNode during classification
"""
cdef struct ObliqueSplitRecord:
    np.ndarray[np.float32_t,ndim=1] hyperplane      #hyperplane defining the split
    double impurity_left                            #impurity of left node
    double impurity_right                           #impurity of right node
    np.npy_int32 split_index                        #The index in instances[] that defines the partition
                                                    #of the instances in the current ObliqueSplit
                                                    #instances[split_index] = an instance in the left partition
                                                    #instances[split_index+1] = an instance in the right partition
