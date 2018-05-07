import numpy as np
from queue import Queue

from ._criterion import get_class_counts

from ._meta import DecisionNode,ObliqueSplitRecord
#from ._splitter import ObliqueSplitter, OC1Splitter, AxisParallelSplitter
"""
Holds classes that build an Oblique Decision Tree
"""

class TreeBuilder:
    def __init__(self, X, y, splitter, min_instances=2):
        """

        :param min_instances: The minimum number of instances required in a DecisionNode to keep splitting. If
                              node.instances_in_node < min_instances the DecisionNode is marked a leaf.
        """

        self.splitter = splitter
        self.X = X
        self.y = y
        self.dimension = X.shape[1] #number of columns
        self.min_instances = min_instances

    def get_majority_class(self, instances, left_boundary_index, right_boundary_index):
        classes = [self.y[instances[i]] for i in range(left_boundary_index, right_boundary_index + 1)]
        values, counts = np.unique(classes, return_counts=True)
       # print(classes)
        #print(values[np.argmax(counts)])
        return values[np.argmax(counts)]

    def build(self):
        """
        Grows an Oblique Decision Tree.
        An array of instances (references to rows of X) is recursively bi-partitioned until
        each partition has cardinality below some threshold (min_instances).
        :return: The root DecisionNode of the DecisionTree
        """


        start = 0
        end= self.X.shape[0]-1
        instances = np.arange(start, end+1, dtype=int) #an array of references to rows in X
        node_queue = Queue()

        root_node = DecisionNode(start, end, 0) #initialize root node
        node_queue.put(root_node)
        splitter = self.splitter

        while(not node_queue.empty()):
            current_node = node_queue.get()

            #label and skip node if it only contains instances from one class.
            class_counts = get_class_counts(instances,current_node.left_boundary_index, current_node.right_boundary_index, self.y)
            if len(class_counts) < 2:
                current_node.is_leaf = True
                current_node.classification = self.get_majority_class(instances, current_node.left_boundary_index,
                                                                      current_node.right_boundary_index)
                continue

            splitter.split(current_node, instances)

            """
            current_node.split_record now contains the split_index that defines the bi-partition of instances in
            instances[] between current_node.left_boundary_index and current_node.right_boundary_index
            """
            if current_node.split_record.get_goodness_of_split() == 0: #mark as leaf node if goodness of split is 0
                current_node.is_leaf = True
                current_node.classification = self.get_majority_class(instances, current_node.left_boundary_index, current_node.right_boundary_index)
                continue

            left_child = DecisionNode(current_node.left_boundary_index,
                                      current_node.split_record.split_index,
                                      current_node.depth + 1)

            right_child = DecisionNode(current_node.split_record.split_index+1,
                                       current_node.right_boundary_index,
                                       current_node.depth + 1)


            current_node.left_child = left_child
            current_node.right_child = right_child



            if left_child.instances_in_node < self.min_instances:
                left_child.is_leaf = True
                left_child.classification = self.get_majority_class(instances, current_node.left_boundary_index,
                                                                      current_node.right_boundary_index)
            else:
                node_queue.put(left_child)

            if right_child.instances_in_node < self.min_instances:
                right_child.is_leaf = True
                right_child.classification = self.get_majority_class(instances, current_node.left_boundary_index,
                                                                      current_node.right_boundary_index)
            else:
                node_queue.put(right_child)

        return root_node
