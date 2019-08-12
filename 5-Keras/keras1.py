# simple linear regression model
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt

# evenly spaced x values
x = data = np.linspace(1, 2, 200)

# generating random y values for above generated x values
# after training, gradient(w) should be around 4.0
y = x*4 + np.random.randn(*x.shape)*0.3

# print(x)
# print()
# print()
# print(y)

model = Sequential()
model.add(Dense(1, input_dim=1, activation='relu'))
model.compile(optimizer="sgd", loss='mse', metrics=['mse'])

# y = w*x + b
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]

print("initial weights:  w = {}     b = {}".format(w_init,b_init))


# batch_size is Number of samples per gradient update
# epoch is an iteration over the entire x and y data provided
model.fit(x,y,batch_size = 1,epochs=20,shuffle=False)
weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]



file_name = "5-Keras/model_for_keras1.py_"

model.save(file_name+".hdf5")

model.save_weights(file_name+"weights.hdf5")



print("trained weights:  w = {}     b = {}".format(w_final,b_final))

#use new weights to predict data
predicted = model.predict(data)

# 'b' means blue line
# 'k.' means black dots
plt.plot(data,predicted,'b',data,y,'k.')
plt.show()
