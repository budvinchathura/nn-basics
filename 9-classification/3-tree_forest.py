#%%
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns


NUM_STEPS = 2000
MINIBATCH_SIZE = 16

#%%

# dataset from https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

file_path_0 = "9-classification/emg_data/0.csv"
file_path_1 = "9-classification/emg_data/1.csv"
file_path_2 = "9-classification/emg_data/2.csv"
file_path_3 = "9-classification/emg_data/3.csv"
df_class_0 = pd.read_csv(file_path_0,header=None)
df_class_1 = pd.read_csv(file_path_1,header=None)
df_class_2 = pd.read_csv(file_path_2,header=None)
df_class_3 = pd.read_csv(file_path_3,header=None)





#%%
emg_df = pd.concat([df_class_0,df_class_1,df_class_2,df_class_3],axis=0,ignore_index=True)
emg_df = emg_df.sample(frac=1).reset_index(drop=True)

print(emg_df.isnull().sum())

#%%

# plot for analysing

# for i in range(65):
#     plt.hist(emg_df[i])
#     plt.title('sensor ' + str(i))
#     plt.show()


#%%

for i in range(64):
    mean_val = emg_df[i].mean()
    std_dev = emg_df[i].std()

    emg_df[i] = emg_df[i].apply(lambda x:(x-mean_val)/std_dev)


#%%
x_input = emg_df.copy()
x_input.drop(64,axis = 1,inplace = True)
y_input = emg_df[64]

x_train,x_test,y_train,y_test = train_test_split(x_input,y_input,test_size = 0.3,random_state = 42)

clrTree = tree.DecisionTreeClassifier()
clrTree.fit(x_train, y_train)


#%%

# Prediction accuracy

treeScore = clrTree.score(x_test,y_test)
treePredict = clrTree.predict(x_test)

print('Desicion Tree Accuracy:',treeScore)
print()

print(classification_report(y_test, treePredict))

print()
print()

randForest = RandomForestClassifier(n_estimators=500,max_features='sqrt')
randForest.fit(x_train, y_train)

forestScore = randForest.score(x_test,y_test)
forestPredict = randForest.predict(x_test)

print('Random Forest Accuracy:',forestScore)
print()

print(classification_report(y_test, forestPredict))




#%%

# cm = confusion_matrix(y_test,treePredict,labels=[0,1,2,3])

# index = [0,1,2,3]  
# columns = [0,1,2,3]  
# cm_df = pd.DataFrame(cm,columns,index)                      
# plt.figure(figsize=(10,6))  
# sns.heatmap(cm_df, annot=True)

# print('Random Forest Accuracy:',forestScore);
