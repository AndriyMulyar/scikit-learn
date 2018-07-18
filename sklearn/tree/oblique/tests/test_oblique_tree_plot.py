from sklearn.tree.oblique.oblique_decision_tree import ObliqueDecisionTree
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=3)


#tree = ObliqueDecisionTree(splitter="axis_parallel", criterion="hellinger", random_seed=5)
tree = DecisionTreeClassifier(criterion="hellinger")
tree.fit(X_train, y_train)

tree.get_decision_boundary_plot(X_train, y_train, f_1=0, f_2 = 1)


