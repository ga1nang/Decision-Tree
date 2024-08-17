from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

#read data
dataset = load_iris()
X = dataset.data
y = dataset.target

#tree model
classifier = tree.DecisionTreeClassifier(criterion='gini',
                                    max_depth=4,
                                    min_samples_leaf=10)
classifier.fit(X, y)

#plot the tree
fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(classifier,ax=ax, feature_names=["sepal length", "sepal width",
                                                "petal length", "petal width"],
                                                 filled=True)
plt.show()