import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus

# sns.set_theme(style='ticks')
# matplotlib.rcParams['figure.dpi'] = 50

df = pd.read_csv('data\\iris_2D.csv')
#sns.pairplot(df, hue='Label')

#load and prepare data
X = df[['Petal_Length', 'Petal_Width']].to_numpy()
X = X.reshape(6, 2)
y = df['Label'].to_numpy()
y = y.reshape(6,)

#DTC 
dt_classifier = DecisionTreeClassifier(max_depth=3,
                                       random_state=1,
                                       criterion='entropy')
dt_classifier.fit(X, y)

#plot the tree
tree.plot_tree(dt_classifier)
plt.show()