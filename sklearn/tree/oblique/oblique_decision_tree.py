import numpy as np, sys
from sklearn.base import BaseEstimator, ClassifierMixin
from mpl_toolkits.axisartist import SubplotZero
from sklearn.utils.validation import check_X_y
from queue import Queue
import matplotlib.pyplot as plt


from ._tree_builder import TreeBuilder
from ._meta import hyperplane_compare
from ._splitter import AxisParallelSplitter, OC1Splitter, AxisParallelDynamicImpuritySplitter
from ._criterion import Gini, Hellinger, Entropy, DynamicImpuritySelection

sys.setrecursionlimit(3000)

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
SPLITTERS = {"axis_parallel": AxisParallelSplitter, "oc1": OC1Splitter, "axis_parallel_dynamic": AxisParallelDynamicImpuritySplitter}
CRITERIONS = {"gini": Gini, "hellinger": Hellinger,"entropy": Entropy, "dynamic": DynamicImpuritySelection}

class ObliqueDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, splitter="oc1", criterion="gini", random_seed = 1, IR_threshold = 60):
        if splitter not in SPLITTERS:
            raise NotImplementedError("No such splitter: {}".format(splitter))
        if criterion not in CRITERIONS:
            raise NotImplementedError("No such criterion: {}".format(criterion))
        self.splitter = splitter
        self.criterion = criterion
        self.random_state = np.random.RandomState(random_seed)
        self.is_fit = False
        self.IR_threshold = IR_threshold

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

        if self.criterion == "dynamic":
            criterion = CRITERIONS[self.criterion](y, imbalance_ratio_threshold=self.IR_threshold)
        else:
            criterion = CRITERIONS[self.criterion](y)

        if self.splitter == "axis_parallel_dynamic":
            splitter = SPLITTERS[self.splitter](X, y, criterion, self.random_state, imbalance_ratio_threshold=self.IR_threshold)
        else:
            splitter = SPLITTERS[self.splitter](X, y, criterion, self.random_state)



        builder = TreeBuilder(X,y,splitter)

        self.tree = builder.build()
        self.is_fit = True

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


    def get_decision_boundary_plot(self, X, y,f_1=0, f_2=1):
        """

        :param f_1: the feature index representing the horizontal axis of the plot
        :param f_2: the feature index representing the vertical axis of the plot
        :return:
        """
        if not self.is_fit:
            raise AssertionError("Tree must be fit to a dataset before it can be drawn.")

        fig = plt.figure(1)
        ax = SubplotZero(fig, 111)
        fig.add_subplot(ax)


        #Plot instances
        markers = {0: '+',
                   1: '^',
                   2: 'x'}
        colors = {'+': 'r',
                  '^': 'b',
                  'x': 'g'}
        for i in range(len(X)):
            plt.plot(X[i][f_1], X[i][f_2] , c=colors[markers[y[i]]],marker=markers[y[i]])


        mins = np.amin(X, axis=0)
        maxs = np.amax(X, axis=0)

        x_min, y_min = mins[f_1], mins[f_2]
        x_max, y_max = maxs[f_1], maxs[f_2]
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        node_queue = Queue()
        node_queue.put(self.tree)

        t = 0
        while(not node_queue.empty()):
            current_node = node_queue.get()
            if current_node.is_leaf:
                continue
            if t==20:
                break
            current_hyperplane = current_node.split_record.hyperplane
            # print(current_hyperplane)

            x = np.linspace(x_min, x_max, 100)
            if current_hyperplane[f_2] == 0:
                y = (-1 * current_hyperplane[f_1] * x - current_hyperplane[-1]) / (current_hyperplane[f_2] + .00001) #add small term to make split not exactly
            else:
                y = (-1 * current_hyperplane[f_1] * x - current_hyperplane[-1]) / (current_hyperplane[f_2])

            # print("x: "+str(x))
            # print("y: "+str(y))
            line = np.polyfit(x, y, 1)  # fit a first degree polynomial to sampled points

            # print("line: "+str(line))
            # print(np.polyval(line, x))
            plt.plot(x, np.polyval(line, x), color='k', linestyle='-')

            node_queue.put(current_node.left_child)
            node_queue.put(current_node.right_child)
            t+=1

        #plt.axis((x_min,x_max,y_min,y_max))
        plt.show()