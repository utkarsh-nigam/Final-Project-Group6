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

import os
os.chdir("/Users/utkarshvirendranigam/Desktop/GWU_course/Machine Learning 1/Project")

data = pd.read_csv("test_data_A-E.csv")

'''
data.rename(columns={'0':'label'}, inplace=True)
filtered_data=data[data["label"]<5]
print(filtered_data.shape)

print(filtered_data.shape)

final_data=pd.DataFrame()

for i in range(0,5):
    tempdf=filtered_data=data[data["label"]==i]
    tempdf=tempdf.sample(frac = 0.25)
    final_data=final_data.append(tempdf)

print(final_data.shape)

final_data.to_csv("test_data_A-E.csv",index=False)











print(data.head())
print('\n\n')
print(data.info())
print('\n\n')
print(data.shape)
print('\n\n')
print(data.label.shape)
'''

# Split data into variables and target
x = data.drop('label',axis = 1)
y = data['label']

# split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=27)

clf = MLPClassifier(hidden_layer_sizes=(80,80), max_iter=200, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


print("Accuarcy: ",accuracy_score(y_test, y_pred))


test = pd.read_csv("predict.csv")
test=test.drop(columns=["Unnamed: 0"])
y_pred = clf.predict(test)
print(y_pred)




'''
arr = test.to_numpy()
print(arr)
new_array=np.abs(arr-255)
print(new_array)

newdf=pd.DataFrame(data=new_array)
print(newdf)

newdf.to_csv("predict.csv", index=False)





new_array=np.reshape(new_array,(28,28))
#print(new_array)
#invert_col_arr = 255 - new_array
plt.imshow(new_array,cmap="gray")
plt.show()

print(new_array)
'''





'''
for i in range(0,12001):
    arr=X_train.iloc[i].to_numpy()
    new_array = np.reshape(arr, (28, 28))
    # print(new_array)
    # invert_col_arr = 255 - new_array
    plt.imshow(new_array, cmap="gray")
    plt.show()
    i+=3000
'''








#cm = confusion_matrix(y_test, y_pred)
#print(cm)

import pickle

filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)
