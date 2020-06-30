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

test = pd.read_csv("predict.csv")
test1=test.drop(columns=["Unnamed: 0","Label"])
y_pred = clf.predict(test1)
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
