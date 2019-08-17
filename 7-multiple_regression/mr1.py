import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as skds
from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
boston = load_boston()
# print(boston.DESCR)

x = boston.data.astype(np.float32)
y = boston.target.astype(np.float32)


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
number_of_epochs = 2000

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
        print("w{} = {:.8f}".format(dim,w_hat[dim][0]))
    
    print()
    for output in range(number_of_outputs):
        print("b{} = {}".format(output,b_hat[output]))
    
    print()
    
    # w_hat = w_hat.reshape(1)

    # print("w = {}      b = {}".format(w_hat[0],b_hat[0]))

    print("mse = {}    r2 = {}".format(mse_score,rsq_score))

    
        
