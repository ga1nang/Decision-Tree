import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus
import dtreeviz

#load dataset
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

#split train test data
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2, random_state=42)

#scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Define an object for DTC and fitting for whole dataset
dt_classifier = DecisionTreeClassifier(max_depth=3,
                                       min_samples_leaf=10,
                                       random_state=1,
                                       criterion='entropy')
dt_classifier.fit(X_train, y_train)

#evaluate the model
y_pred = dt_classifier.predict(X_test)
print('Accaracy: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

tree.plot_tree(dt_classifier, filled=True)
plt.show()

#render as SVG
viz_model = dtreeviz.model(dt_classifier,
                           X_train=X_train, y_train=y_train,
                           feature_names=iris.feature_names,
                           target_name='iris',
                           class_names=iris.target_names)

v = viz_model.view() #render as SVG into internal object
v.show() #pop up window