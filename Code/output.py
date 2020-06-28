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

for i in range(0,len(y_pred)):
    print("Actual: ",test["Label"].iloc[i]," Predicted: ", folders[y_pred[i]])