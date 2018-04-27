import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y

from ._tree_builder import TreeBuilder
from ._meta import hyperplane_compare
from ._splitter import AxisParallelSplitter, OC1Splitter
from ._criterion import Gini, Hellinger
"""
Oblique Decision Trees partition the decision space using a linear combination of features
as opposed to traditional single feature, axis parallel splits.

Oblique splits can be geometrically interpreted as oblique hyperplanes that bi-partition the decision space.
Notice that traditional axis parallel splitting is the degenerate case of oblique splitting 
(hyperplane has all zero coefficients except the feature being split on).
"""


"""
Define splitters and criteria
"""
SPLITTERS = {"axis_parallel": AxisParallelSplitter, "oc1": OC1Splitter}
CRITERIONS = {"gini": Gini, "hellinger": Hellinger}

class ObliqueDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, splitter="oc1", criterion="gini", random_seed = 1):
        if splitter not in SPLITTERS:
            raise NotImplementedError("No such splitter: {}".format(splitter))
        if criterion not in CRITERIONS:
            raise NotImplementedError("No such criterion: {}".format(criterion))
        self.splitter = splitter
        self.criterion = criterion
        self.random_state = np.random.RandomState(random_seed)

    def fit(self, X, y):
        """
        Fits an Oblique Decision tree on X,y

        An array of instances serving as pointers to the rows in X is created.
        This array is divided into two parts representing a binary split of the instances.
        Recursively, each part is again sub-divided until conditions for a leaf node are satisfied.

        :param X: Instance matrix
        :param y: Class labels
        :param splitter: type of splitter (axis_parallel, oc1)
        :param criterion: type of splitting criterion (gini, hellinger)
        :return:
        """
        splitter = SPLITTERS[self.splitter](X, y, CRITERIONS[self.criterion](y), self.random_state)

        builder = TreeBuilder(X,y,splitter)

        self.tree = builder.build()

    def predict(self, X):
        """
        Predicts a given instance by traversing the nodes of a trained Oblique Decision Tree
        :param X:
        :return: Class label of X
        """
        y = []
        for x in X:
            current_node = self.tree
            while not current_node.is_leaf:
                if hyperplane_compare(x, current_node.split_record.hyperplane) <= 0: # below the hyperplane
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child
            y.append(current_node.classification)
        return y