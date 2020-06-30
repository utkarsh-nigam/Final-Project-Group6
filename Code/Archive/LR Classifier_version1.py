import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("final_data_a_z.csv")
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
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# Logistics Model
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)

# for one observation
#predictions = logisticRegr.predict(X_test[0].reshape(1,-1))
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)
print(score)
print('\n\n')

print("CM")
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
plt.show()
print("plot")


print("Classification Report: ")
print(classification_report(y_train,y_test))
print("\n")

print("Accuracy : ", accuracy_score(y_train,y_test) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_train,y_test[:,1]) * 100)
