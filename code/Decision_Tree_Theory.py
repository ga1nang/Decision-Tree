import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#A figure is created to show Gini impurity measures
plt.figure()
x = np.linspace(0.01, 1)
y = 1 - (x*x) - (1-x)*(1-x)
plt.plot(x, y)
plt.title('Gini Impurity')
plt.xlabel('Fraction of Class k ($p_k$)')
plt.ylabel('Impurity Measures')
plt.xticks(np.arange(0, 1.1, 0.1))

#plt.show()

#Defining a simple dataset
attribute_names = ['age', 'income','student', 'credit_rate']
class_name = 'default'
data1 ={
    'age' : ['youth', 'youth', 'middle_age', 'senior', 'senior', 'senior','middle_age', 'youth', 'youth', 'senior', 'youth', 'middle_age','middle_age', 'senior'],
    'income' : ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium','low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student' : ['no','no','no','no','yes','yes','yes','no','yes','yes','yes','no','yes','no'],
    'credit_rate' : ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair','excellent', 'excellent', 'fair', 'excellent'],
    'default' : ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes','yes', 'yes', 'yes', 'no']
}
df1 = pd.DataFrame(data1, columns=data1.keys())
print(df1)

#calculate gini(D)
def gini_impurity(value_counts):
    n = value_counts.sum()
    p_sum = 0
    for key in value_counts.keys():
        p_sum += (value_counts[key] / n) * (value_counts[key] / n)
    gini = 1 - p_sum
    return gini

class_value_counts = df1[class_name].value_counts()
print(f'Number of samples in each class is:\n{class_value_counts}')

gini_class = gini_impurity(class_value_counts)
print(f'\nGini Impurity of the class is {gini_class:.3f}')

#calculate gini impurity for the attributes
def gini_split_a(attribute_name):
    attribute_values = df1[attribute_name].value_counts()
    gini_A = 0
    for key in attribute_values.keys():
        df_k = df1[class_name][df1[attribute_name] == key].value_counts()
        n_k = attribute_values[key]
        n = df1.shape[0]
        gini_A += ((n_k / n) * gini_impurity(df_k))
    return gini_A

gini_attribute = {}
for key in attribute_names:
    gini_attribute[key] = gini_split_a(key)
    print(f'Gini for {key} is {gini_attribute[key]:.3f}')
    
#compute gini gain values to find the best split
#attribute has maximum Gini gain is selected for splitting

min_value = min(gini_attribute.values())
print('The minimum value of Gini Impurity : {0:.3}'.format(min_value))
print('The maximum value of Gini Gain : {0:.3}'.format(1 - min_value))

selected_attribute = min(gini_attribute.keys())
print('The selected attribute is: ', selected_attribute)