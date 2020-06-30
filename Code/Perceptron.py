import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score
from sklearn.metrics import accuracy_score
# import os
# os.chdir("C:\Users\nsree_000\Desktop\Python-Quiz\Local-Final-Project-Group6\Code\Saved Models")

import pickle
import warnings

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

data = pd.read_csv("final_data_a_z.csv")
data.rename(columns={'0':'label'}, inplace=True)

print(data.head())
print('\n\n')
print(data.info())
print('\n\n')
print(data.shape)
print('\n\n')
print(data.label.shape)

# Split data into variables and target
X = data.drop('label',axis = 1)
y = data['label']
#-----------------------------------------------------------------------------
# data pre processing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)
#-----------------------------------------------------------------------------

# machine learning
from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=30, eta0=0.1, random_state=0)

Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=0.1,
           fit_intercept=True, max_iter=10, n_iter_no_change=5, n_jobs=None,
           penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=0, warm_start=False)
ppn.fit(X_train, y_train)

y_pred = ppn.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

from sklearn.metrics import recall_score
print('Recall_Score: %.2f' % recall_score(y_test, y_pred, pos_label='positive', average='micro'))

from sklearn.metrics import precision_score
print('Precision_Score: %.2f' % precision_score(y_test, y_pred, pos_label= 1 , average='micro'))

from sklearn.metrics import f1_score
print('F1_score: %.2f' % f1_score(y_test, y_pred,pos_label= 1 , average='micro'))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

test = pd.read_csv("predict.csv")
test.dropna(subset=["Label"],inplace = True)
test1 = test.drop(columns=["Label"])

y_pred = ppn.predict(test1)
folders=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
y_pred = ppn.predict(test1)
print(y_pred)

# for i in range(0,len(y_pred)):
#     print("Actual: ",test["Label"].iloc[i]," Predicted: ", folders[y_pred[i]])
#
# filename = 'PPN-Final.sav'
# ppn = pickle.load(open(filename, 'rb'))
# pickle.dump(ppn, open(filename, 'wb'))
# loaded_model = pickle.load(open(filename, 'rb'))
# #-----------------------------------------------------------------------------
# # ppn = pickle.load(open('Saved Models\LR_predict2.sav','rb'))
#
# test = pd.read_csv("predict_New.csv")
# test1 = test.drop(columns=["Label"])
# y_pred = ppn.predict(test1)
# folders=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
#
attempts=np.zeros(26,int)
correct=np.zeros(26,int)

attempt_dict=dict(zip(folders,attempts))
correct_dict=dict(zip(folders,correct))
missclassify_dict=dict()

for i in range(0,len(y_pred)):
    print("Actual: ",test["Label"].iloc[i]," Predicted: ", folders[y_pred[i]])
    attempt_dict[test["Label"].iloc[i]]+=1
    if(test["Label"].iloc[i]==folders[y_pred[i]]):
        correct_dict[test["Label"].iloc[i]] += 1
    else:
        if test["Label"].iloc[i] in missclassify_dict.keys():
            missclassify_dict[test["Label"].iloc[i]].append(folders[y_pred[i]])

        else:
            missclassify_dict[test["Label"].iloc[i]]=[folders[y_pred[i]]]

for key in attempt_dict:
    if (attempt_dict[key]>0):
        print("\nCharacter: ", key, "\tAttempts: ",attempt_dict[key], "\tCorrect: ",correct_dict[key], "\tAccuracy: ", round((correct_dict[key]/attempt_dict[key])*100,1),end="\t")
        if key in missclassify_dict.keys():
            print("\tMissclassified with: ", missclassify_dict[key])
        else:
            print("Missclassified with: ", "None")
# #-----------------------------------------------------------------------------
