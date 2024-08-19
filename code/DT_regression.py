import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import dtreeviz
import matplotlib

#load and plot
df = pd.read_csv('data\\Salary_Data_simple.csv')
#plt.scatter(x=df['Experience'], y=df['Salary'])
#plt.show()

#prepare data for training
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

#DT regression model
regressor = DecisionTreeRegressor(max_depth=2)
regressor.fit(X_train, y_train)

#result
y_pred = regressor.predict(X_test)
print(r2_score(y_test, y_pred))

#text representation
text_representation = tree.export_text(regressor)
print(text_representation)

#plot the tree
fig = plt.figure(figsize=(15, 8))
tree.plot_tree(regressor, feature_names=['YearsExperience'],
               filled=True)
plt.show()

#using dtreeviz to plot
viz = dtreeviz.model(regressor, X_train, y_train,
                target_name="target",
                feature_names=['YearsExperience'])
v = viz.view()     # render as SVG into internal object 
v.show()              # pop up window