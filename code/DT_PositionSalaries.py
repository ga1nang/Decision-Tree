import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import tree

#load and prepare data
dataset = pd.read_csv('data\\Position_Salaries.csv')
X = dataset.iloc[:, 1:2]
y = dataset.iloc[:, -1]

#DT with max_depth 3
dt_max_depth_3 = DecisionTreeRegressor(random_state=0, max_depth=3)
dt_max_depth_3.fit(X, y)
fig, ax = plt.subplots(figsize=(10, 8))
tree.plot_tree(dt_max_depth_3, feature_names=['Level'], filled=True)
plt.show()

#DT with min samples leaf 4
dt_min_samples_leaf_4 = DecisionTreeRegressor(random_state=0, min_samples_leaf=4)
dt_min_samples_leaf_4.fit(X, y)


#export the dt to a tree.dot file
export_graphviz(dt_max_depth_3, out_file='tree_graph\\tree.dot', feature_names=['Level'])


#compare two DT models
plt.figure()
plt.scatter(X, y, marker="x", color='red', label='Data')
plt.plot(X, dt_max_depth_3.predict(X), color='blue', label='max_depth_3')
plt.plot(X, dt_min_samples_leaf_4.predict(X), marker='D', color='green', label='min sample leaf = 4')
plt.title('Compare 2 Decision Tree model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()