# simple linear regression model
# using gradient descent

from sklearn import datasets as skds
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# create dummy data
x,y = skds.make_regression(n_samples=200,n_features=1,n_informative=1,n_targets=1,noise=15.0)

# print(y)

# reshape numpy array to have two dimensions
if(y.ndim == 1):
    y = y.reshape(len(y),1)

print()


# get 30% of data for testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=123)

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

w_hat = 0
b_hat = 0
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
    
    w_hat = w_hat.reshape(1)

    print("w = {}      b = {}".format(w_hat[0],b_hat[0]))

    print("mse = {}    r2 = {}".format(mse_score,rsq_score))
    
    y_predict = x*w_hat + b_hat
    plot.plot(x,y,'.b')
    plot.plot(x,y_predict,'r-')
    plot.show()










