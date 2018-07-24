"""
The prototype version of the Branching Tree Ensemble.

A branching tree ensemble grows learners on the branches of an induced decision tree
"""

from .base import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
import collections
import numpy as np

class BranchingTreeEnsemble(BaseEnsemble):

    def __init__(self, architect_tree, base_estimator,  combination_method="majority_vote", depth_first=False, use_architect=False, random_state=None):
        """

        :param architect_tree: The DecisionTree that will architect the ensemble structure
        :param base_estimator: The base model that will be induced on the architecture
        :param combination_method: Combination method to utilize (majority_vote, weighted_majority_vote
        :param depth_first: if true only consider classifiers along architects classification path
        :param random_state:
        """
        if not isinstance(architect_tree, DecisionTreeClassifier):
            raise Exception("Must provide a DecisionTreeClassifier as the architect")

        self.architect_tree = architect_tree
        self.base_estimator = base_estimator
        self.estimator_params = tuple()
        self.random_state = random_state
        self.method = combination_method
        self.depth_first = depth_first
        self.use_architect = use_architect
        self.architect_tree.random_state = random_state




    #Retrieve all samples reaching a given node
    # https://stackoverflow.com/questions/45398737/is-there-any-way-to-get-samples-under-each-leaf-of-a-decision-tree


    def fit(self, X,y):
        """
        Fits a Branching Tree Ensemble on X,y
        1) Constructs architect tree and stores references to samples in node i inside samples[i]
           These samples are accessible using X[samples[i]], y[samples[i]]

        2) Trains a base learner on samples at each non-leaf node
        :param X:
        :param y:
        :return:
        """
        architect = self.architect_tree

        architect.fit(X,y)

        n_nodes = architect.tree_.node_count
        children_left = architect.tree_.children_left
        children_right = architect.tree_.children_right
        impurity = architect.tree_.impurity

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64) #holds the depth of node i
        is_leaf = np.zeros(shape=n_nodes, dtype=bool)      #holds whether node i is a leaf

        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaf[node_id] = True

        number_non_leaf_nodes = (is_leaf == False).sum()

        #Initialize BaseEnsemble
        super().__init__(base_estimator=self.base_estimator, n_estimators=number_non_leaf_nodes, estimator_params=self.estimator_params)
        self._validate_estimator()

        #Retrieve instances in nodes
        #X[samples[i]] holds the samples reaching node i
        samples = collections.defaultdict(list)
        dec_paths = architect.decision_path(X)
        for d, dec in enumerate(dec_paths):
            for i in range(architect.tree_.node_count):
                if dec.toarray()[0][i] == 1:
                    samples[i].append(d)

        # for x in samples.keys():
        #     print( x,",", is_leaves[x] ,  samples[x])
        # print(number_non_leaf_nodes)

        self.estimators_ = [] #holds all base learners
        self.estimator_to_node_map = {} #maps indices in self.estimators to tree node indices
        self.node_to_estimator_map = {}

        #we begin by fitting estimators on all decision nodes for testing
        for i in range(n_nodes):
            if not is_leaf[i]:
                #print("Creating %i" % i)
                self.estimator_to_node_map[len(self.estimators_)] = i
                self.node_to_estimator_map[i] = len(self.estimators_)
                estimator = self._make_estimator(append = True, random_state=self.random_state)
                estimator.fit(X[samples[i]], y[samples[i]])
                #print("Fit %i" % i)
        #print(self.node_to_estimator_map)

    def predict_architect_(self, X):
        return self.architect_tree.predict(X)

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        all_predictions = [clf.predict(X) for clf in self.estimators_]
        if self.depth_first:
            all_predictions = []
            dec_paths = self.architect_tree.decision_path(X)
            for sample_index in range(len(X)):
                node_indices = dec_paths.indices[dec_paths.indptr[sample_index]:dec_paths.indptr[sample_index + 1]]
                estimator_indices = []
                for node_index in node_indices:
                    if node_index in self.node_to_estimator_map: #if key does not exist, node is a leaf node
                        estimator_indices.append(self.node_to_estimator_map[node_index]) #add estimator indicies for sample

                # print(sample_index)
                # print(node_indices)
                # print(estimator_indices)
                estimators_on_path = [self.estimators_[i] for i in estimator_indices]
                sample_predications = [clf.predict([X[sample_index]])[0] for clf in estimators_on_path]
                if self.use_architect:
                    sample_predications.append(self.architect_tree.predict([X[sample_index]])[0])
                all_predictions.append(np.asarray(sample_predications,dtype='int64'))
            return np.asarray(all_predictions).T
        else:
            all_predictions = [clf.predict(X) for clf in self.estimators_]
            if self.use_architect:
                all_predictions.append(self.predict_architect_(X))
            return np.asarray(all_predictions).T


    def predict(self, X):
        """
        :param X: array of test samples
        :param method: majority_vote, weighted_majority_vote
        :param depth_first: if true only consider classifiers along architects classification path
        :return:
        """
        if self.method == "majority_vote":
            return self.majority_vote_prediction_(X)

        if self.method == "weighted_majority_vote":
            return self.weighted_majority_vote_prediction_(X)

    def majority_vote_prediction_(self, X):
        predictions = self._predict(X)
        #print(predictions)
        if self.depth_first:
            maj = [np.argmax(np.bincount(x)) for x in predictions]
        else:
            maj = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=None)),
                axis=1, arr=predictions)

        return maj


    def weighted_majority_vote_prediction_(self, X):
        pass













