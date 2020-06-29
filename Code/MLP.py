import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir("/Users/utkarshvirendranigam/Desktop/GWU_course/Machine Learning 1/Project")

data = pd.read_csv("A_Z Handwritten Data.csv")
data.rename(columns={'0':'label'}, inplace=True)
print(data.columns)

'''
data = pd.read_csv("A_Z Handwritten Data.csv")
data.rename(columns={'0':'label'}, inplace=True)
#filtered_data=data[data["label"]<5]
#print(filtered_data.shape)

#print(filtered_data.shape)

final_data=pd.DataFrame()
letter_selected=[1,7,3,5,15,8,9,19,12,13,10,14,17,21,20,22]
for i in letter_selected:
    tempdf=data[data["label"]==i]
    #tempdf=tempdf.sample(frac = 0.25)
    final_data=final_data.append(tempdf)

print(final_data.shape)

final_data.to_csv("Selected_Data.csv",index=False)

'''

# Split data into variables and target
x = data.drop('label',axis = 1)
y = data['label']

# split the data
'''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.005, random_state=27)



clf = MLPClassifier(hidden_layer_sizes=(80,80,80),activation="tanh", max_iter=200, alpha=0.0001,
                     solver='adam', verbose=10,  random_state=21, tol=0.000000001)

clf.fit(x, y)

y_pred = clf.predict(X_test)


print("Accuarcy: ",accuracy_score(y_test, y_pred))


filename = 'MLP_Adam_80_SelectedData_tanh.sav'
pickle.dump(clf, open(filename, 'wb'))
'''




clf = MLPClassifier(hidden_layer_sizes=(100,100,100),activation="logistic", max_iter=200, alpha=0.0001,
                     solver='adam', verbose=10,  random_state=21, tol=0.000000001,early_stopping=True)


clf.fit(x, y)


#y_pred = clf.predict(X_test)


#print("Accuarcy: ",accuracy_score(y_test, y_pred))


filename = 'MLP_Adam_100_AllData_sigmoid_earlystop.sav'
pickle.dump(clf, open(filename, 'wb'))










filename = 'MLP_Adam_100_SelectedData_sigmoid_earlystop.sav'
clf3 = pickle.load(open(filename, 'rb'))


test = pd.read_csv("predict 5.csv")
test1=test.drop(columns=["Label"])
y_pred = clf.predict(test1)

folders=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

attempts=np.zeros(26,int)
correct=np.zeros(26,int)

attempt_dict=dict(zip(folders,attempts))
correct_dict=dict(zip(folders,correct))


missclassify_dict=dict()

plot_letter=["B","M","E","N","I","W","R"]

for i in range(0,len(y_pred)):
    print("Actual: ",test["Label"].iloc[i]," Predicted: ", folders[y_pred[i]])
    attempt_dict[test["Label"].iloc[i]]+=1
    if(test["Label"].iloc[i]==folders[y_pred[i]]):
        correct_dict[test["Label"].iloc[i]] += 1
    else:
        if (test["Label"].iloc[i] in plot_letter):
            array = test1.iloc[i].to_numpy()
            new_array = np.reshape(array, (28, 28))
            plt.imshow(new_array, cmap="gray")
            plt.show()

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




filename = 'MLP_Adam_80_CompleteData.sav'
loaded_model = pickle.load(open(filename, 'rb'))

test = pd.read_csv("predict 3.csv")
test1=test.drop(columns=["Unnamed: 0","Label"])
y_pred = loaded_model.predict(test1)
folders=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]






for i in range(0,len(y_pred)):
    print("Actual: ",test["Label"].iloc[i]," Predicted: ", folders[y_pred[i]])
    array=test1.iloc[i].to_numpy()
    new_array = np.reshape(array, (28, 28))
    # print(new_array)
    # invert_col_arr = 255 - new_array
    plt.imshow(new_array, cmap="gray")
    plt.show()

    #print(new_array)












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

