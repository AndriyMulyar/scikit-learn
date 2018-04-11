import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
"""
Oblique Decision Trees partition the decision space using a linear combination of features
as opposed to the traditional single feature, axis parallel splits.

The splits can be geometrically interpreted as oblique hyperplanes that bi-partition the decision space.
Notice that traditional axis parallel splitting is the degenerate case of oblique splitting 
(hyperplane has all zero coefficients except the feature being split on).
"""


"""
Define splitters and criteria
"""


class ObliqueDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fits an Oblique Decision tree on X,y

        An array of instances serving as pointers to the rows in X is created.
        This array is divided into two parts representing a binary split of the instances.
        Recursively, each part is again sub-divided until conditions for a leaf node are satisfied.

        :param X: Instance matrix
        :param y: Class labels
        :return:
        """




        pass

    def predict(self, X):
        """
        Predicts a given instance by traversing the nodes of a trained Oblique Decision Tree
        :param X:
        :return: Class label of X
        """
        pass