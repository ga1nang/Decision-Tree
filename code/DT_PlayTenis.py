import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import dtreeviz

#load data
df = pd.read_csv('data\\Play Tennis.csv')


#preprocessing data 
string_to_int = preprocessing.LabelEncoder()
df = df.apply(string_to_int.fit_transform)
print(df)


#prepare data for training
features_cols = ['Outlook','Temprature','Humidity','Wind']
X = df[features_cols]
y = df.Play_Tennis


#Training and plot the tree
classifier = DecisionTreeClassifier(criterion='entropy', random_state=100)
classifier.fit(X, y)
plt.figure(figsize=(15, 6))
plot_tree(classifier, filled=True, feature_names=features_cols, class_names=['yes', 'no'], rounded=True)
#plt.show()


#viz model
viz_model = dtreeviz.model(classifier,
                           X_train=X, y_train=y,
                           feature_names=['Outlook','Temprature','Humidity','Wind'],
                           target_name='Play_Tennis')

v = viz_model.view()
v.show()