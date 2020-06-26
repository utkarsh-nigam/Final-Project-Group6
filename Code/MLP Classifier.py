import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("A_Z Handwritten Data.csv")
data.rename(columns={'0':'label'}, inplace=True)
print(data.head())
print('\n\n')
print(data.info())
print('\n\n')
print(data.shape)
print('\n\n')
print(data.label.shape)

# Split data into variables and target
x = data.drop('label',axis = 1)
y = data['label']

# split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, center=True)
plt.show()

