import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
import pickle

import warnings
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
x = data.drop('label',axis = 1)
y = data['label']

# split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Logistics Model
logisticRegr = LogisticRegression(solver ='lbfgs')
logisticRegr.fit(X_train, y_train)


y_pred = logisticRegr.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('\n')
print('Recall_Score: %.2f' % recall_score(y_test, y_pred, pos_label='positive', average='micro'))
print('\n')
print('Precision_Score: %.2f' % precision_score(y_test, y_pred, pos_label= 1 , average='micro'))
print('\n')
print('F1_score: %.2f' % f1_score(y_test, y_pred,pos_label= 1 , average='micro'))
print('\n')
print(classification_report(y_test, y_pred))
print('\n')

test = pd.read_csv("predict.csv")
test.dropna(subset=["Label"],inplace = True)
test1=test.drop(columns=["Label"])
y_pred = logisticRegr.predict(test1)
folders=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

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


