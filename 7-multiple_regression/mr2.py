import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as skds
from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys


import pandas as pd
filePath = "7-multiple_regression/student_results.csv"


df = pd.read_csv(filePath)
print(df.head())

# print(df.columns)

df.drop(['GPA'],axis=1,inplace=True)


# print(df.columns)

# df.iloc[df.isnull()] = 0

# print(df.isnull().sum())

# MT = 0
# CH = 1
# ME = 2
# EE = 3
# CE = 4
# CS = 5
# BM = 6
# EN = 7

fields = ['MT','CH','ME','EE','CE','CS','BM','EN']

for i in range(len(fields)):
    df['Field'].replace(fields[i],i,inplace=True)


df.loc[df['Field'].isnull(),'Field'] = 0

if(sorted(df['Field'].unique())==[0,1,2,3,4,5,6,7]):
    print("Field column pre processing done...")
else:
    print("Field column preprocessing error")
    sys.exit()

grades = {'A+':4.2,'A':4.0,'A-':3.7,'B+':3.3,'B':3.0,'B-':2.7,'C+':2.3,'C':2.0,'C-':1.5,'D':1.0,'F':0.0,'IWE':0.0,'ICA':0.0,'I-we':0.0}

for key in grades.keys():
    df.replace(key,grades[key],inplace=True)


modules = list(df.columns)
modules.remove('Field')
modules.remove('Rank')

# print(modules)

grades_set = set(grades.values())

for module in modules:
    df.loc[df[module].isnull(),module] = 0.0
    # print(set(df[module].unique()).issubset(grades_set))
    if(not(set(df[module].unique()).issubset(grades_set))):
        print("module columns preprocessing error")
        sys.exit()
        break
else:
    print("module columns preprocessing done...")


largest_rank = df["Rank"].max()
print(largest_rank)
df["Rank"] = df["Rank"].apply(lambda x: largest_rank - x+1)


if(set(df["Rank"].unique()).issubset(set(list(range(0,largest_rank+1))))):
    print("Rank column preprocessing done")
else:
    print("Rank column preprocessing error")
    sys.exit()

df["Rank"] = df["Rank"].apply(lambda x: 10*x/float(largest_rank))

# print(df["Rank"])


# print(df)

# print(df.values)

df_copy = df.copy(deep=True)

y = df_copy['Rank'].to_numpy()
df_copy.drop(['Field','Rank'],axis=1,inplace=True)


x = df_copy.values

# print(x)
# print(y)


if(y.ndim == 1):
    # when -1 is used that dimension will be set automatically
    y=y.reshape(-1,1) 

# remove mean and scale to unit variance
# each feature is scaled seperately          
x = StandardScaler().fit_transform(x)


# 30% of data for training
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=123)

# print(x_train.shape)

number_of_inputs = x_train.shape[1]
number_of_outputs = y_train.shape[1]

# using placeholders to store inputs

x_tensor = tf.placeholder(dtype = tf.float32,shape=[None,number_of_inputs],name="tensor_x")   #None size enables auto dimension size

y_tensor = tf.placeholder(dtype = tf.float32,shape=[None,number_of_outputs],name = "tensor_y")

w = tf.Variable(tf.zeros([number_of_inputs,number_of_outputs]),dtype=tf.float32,name="w")
b = tf.Variable(tf.zeros([number_of_outputs]),dtype=tf.float32,name="b")

model = tf.matmul(x_tensor,w) + b

# calculating loss
loss = tf.reduce_mean(tf.square(model-y_tensor))

# calculating mean sqrd error
mse = tf.reduce_mean(tf.square(model-y_tensor))

y_mean = tf.reduce_mean(y_tensor)

total_error = tf.reduce_sum(tf.square(y_tensor-y_mean))

u_error = tf.reduce_sum(tf.square(y_tensor-model))

rsq = 1-tf.divide(u_error,total_error)

# define optimizer function

learning_rate = 0.001

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# train the model
number_of_epochs = 3000

w_hat = []
b_hat = []
loss_epochs = np.empty(shape=[number_of_epochs],dtype=float)
mse_epochs = np.empty(shape=[number_of_epochs],dtype=float)
rs_epochs = np.empty(shape=[number_of_epochs],dtype=float)

# intial values
mse_score = 0
rsq_score = 0

with tf.Session() as tfs:
    # initialize global variables
    tfs.run(tf.global_variables_initializer())

    for epoch in range(number_of_epochs):
        
        # train with train data
        feed_dict = {x_tensor:x_train,y_tensor:y_train}
        loss_val,_ = tfs.run([loss,optimizer],feed_dict = feed_dict)
        # store loss value
        loss_epochs[epoch] = loss_val


        # evaluate with test data
        feed_dict = {x_tensor:x_test,y_tensor:y_test}
        mse_score,rsq_score = tfs.run([loss,rsq],feed_dict=feed_dict)

        mse_epochs[epoch] = mse_score
        rs_epochs[epoch] = rsq_score
        print("epoch:{}   loss = {:.4f}     mse = {:.4f}     r_sqrd = {:.4f}".format(epoch,loss_val,mse_score,rsq_score))
    
    w_hat,b_hat = tfs.run([w,b])

    print()
    for dim in range(number_of_inputs):
        print("{} = {:.8f}".format(modules[dim],w_hat[dim][0]))
    
    print()
    for output in range(number_of_outputs):
        print("b{} = {}".format(output,b_hat[output]))
    
    print()
    
    # w_hat = w_hat.reshape(1)

    # print("w = {}      b = {}".format(w_hat[0],b_hat[0]))

    print("mse = {}    r2 = {}".format(mse_score,rsq_score))
