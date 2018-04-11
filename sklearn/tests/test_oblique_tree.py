from sklearn.tree.oblique.oblique_decision_tree import ObliqueDecisionTree
from sklearn import datasets

iris = datasets.load_iris()


tree = ObliqueDecisionTree()
tree.fit(iris.data, iris.target)