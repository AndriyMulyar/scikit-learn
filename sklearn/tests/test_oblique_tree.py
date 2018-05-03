from sklearn.tree.oblique.oblique_decision_tree import ObliqueDecisionTree
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=3)
#print(X_train)
tree_gini = ObliqueDecisionTree(splitter="oc1", criterion="gini", random_seed=5)
tree_gini.fit(X_train,y_train)

#tree_hellinger = ObliqueDecisionTree(splitter="oc1", criterion="hellinger", random_seed=5)
#tree_hellinger.fit(X_train, y_train)

#print(X_test)

print(tree_gini.score(X_test,y_test))
#print(tree_hellinger.score(X_test,y_test))


# tree_gini = ObliqueDecisionTree(splitter="oc1", criterion="gini", random_seed=5)
# tree_gini.fit(iris.data,iris.target)
