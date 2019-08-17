import keras

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# data set from https://www.kaggle.com/testpython/linear-regression-with-single-variable
train_file_path = "5-Keras/train_for_keras2.py_.csv"

train_data = pd.read_csv(train_file_path)
print(train_data.head())


# pre processing
mean_x = train_data['x'].mean(skipna=True)
mean_y = train_data['y'].mean(skipna=True)

train_data.loc[train_data['x'].isnull(),'x'] = mean_x
train_data.loc[train_data['y'].isnull(),'y'] = mean_y

train_data.drop_duplicates(['x'],keep='first',inplace=True)
train_data.sort_values(by=['x'],inplace=True)

train_data = train_data[(train_data['y']>0) & (train_data['x']<300)]
# train_data = train_data[(train_data['y']>0) & (train_data['x']>10)  ]

print(train_data)




train_x = pd.Series(train_data['x']).to_numpy(dtype="float32")
train_y = pd.Series(train_data['y']).to_numpy(dtype="float32")

# print(train_x)
# print(train_y)

model = Sequential()
model.add(Dense(1, input_dim=1, activation='relu'))
# sgd_opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer="Adam", loss='mse', metrics=['mse'])

# y = w*x + b
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]

print("initial weights:  w = {}     b = {}".format(w_init,b_init))

# print(train_y)

# batch_size is Number of samples per gradient update
# epoch is an iteration over the entire x and y data provided
model.fit(train_x,train_y,batch_size = 1,epochs=50,shuffle=False)
weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]



# file_name = "5-Keras/model_for_keras1.py_"

# model.save(file_name+".hdf5")

# model.save_weights(file_name+"weights.hdf5")



print("trained weights:  w = {}     b = {}".format(w_final,b_final))

# use new weights to predict data
predicted = model.predict(train_x)

# 'b' means blue line
# 'k.' means black dots
plt.plot(train_x,predicted,'b',train_x,train_y,'k.')
# plt.plot(train_x,train_y,'k.')
plt.show()