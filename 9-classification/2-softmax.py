#%%
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# print(df_class_0.count());
# print(df_class_1.count());
# print(df_class_2.count());
# print(df_class_3.count());



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
class_0 = np.asarray([1,0,0,0])
class_1 = np.asarray([0,1,0,0])
class_2 = np.asarray([0,0,1,0])
class_3 = np.asarray([0,0,0,1])
emg_df[64] = emg_df[64].map({0:class_0,1:class_1,2:class_2,3:class_3})

#%%
x_input = emg_df.copy()
x_input.drop(64,axis = 1,inplace = True)
y_input = emg_df[64]

x_train,x_test,y_train,y_test = train_test_split(x_input,y_input,test_size = 0.3,random_state = 42)

x_input = x_train
y_input = y_train

#%%
# placeholder and vars
x = tf.placeholder(tf.float32,shape = [None,64])
y_ = tf.placeholder(tf.float32,shape = [None,4])

# weights and biases
w = tf.Variable(tf.ones([64,4]))
b = tf.Variable(tf.ones([4]))

#%%

# softmax function
y = tf.nn.softmax(tf.matmul(x,w) + b)

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

#optimizer

train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

# accuracy
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#%%
# session
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
epoch = 5000

#%%
for step in range(epoch):
    _, c = sess.run([train_step,cross_entropy],feed_dict={x:x_input,y_:[t for t in y_input.to_numpy()]})
    if(step%100 == 0):
        print("step: "+str(step)+"   c: "+str(c))


print()
print('accuracy =',sess.run(accuracy,feed_dict={x:x_test,y_:[t for t in y_test.to_numpy()]}))