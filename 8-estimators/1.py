#%%
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import datasets, metrics, preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



#%%

boston_data = datasets.load_boston()

boston_df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston_df['MEDV'] = boston_data.target


# print(boston_df.isnull().sum())
# x_data = preprocessing.StandardScaler().fit_transform(boston.data)
# y_data = boston.target

NUM_STEPS = 2000
MINIBATCH_SIZE = 16

# boston_df["INDUS"].plot(kind='hist', grid=True)
# plt.show()

#%%
# CHAS categorical feature need string or int values
boston_df['CHAS'] = boston_df['CHAS'].astype('int64')

features = boston_df.drop('MEDV', axis=1)
target = boston_df['MEDV']


#%%
# 30% for testing
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=55)

# seperating feature columns

all_columns = list(features.columns)

numeric_columns = all_columns[::]
numeric_columns.remove('CHAS')

categorical_columns = ['CHAS']

numeric_features = [tf.feature_column.numeric_column(
    key=column) for column in numeric_columns]

# 'CHAS' column has only 0,1 values
categorical_features = [tf.feature_column.categorical_column_with_vocabulary_list(
    key=column, vocabulary_list=features[column].unique()) for column in categorical_columns]


linear_features = numeric_features + categorical_features

#%%
# use builtin input functions

# shuffle parameter is compulsory
input_fn_train = tf.estimator.inputs.pandas_input_fn(
    x=x_train, y=y_train, batch_size=MINIBATCH_SIZE, num_epochs=None,shuffle=True)

input_fn_eval = tf.estimator.inputs.pandas_input_fn(x = x_test,y=y_test,batch_size = MINIBATCH_SIZE,num_epochs = 1,shuffle = False)

#%%
# instantiate and run model
# model_dir location to save the model

# tf.reset_default_graph()

linear_regressor = tf.estimator.LinearRegressor(feature_columns = linear_features,model_dir = "8-estimators/save_data_for_1.py_2")
linear_regressor.train(input_fn = input_fn_train,steps = NUM_STEPS)

#%%
# evaluate with test data
metrics_data = linear_regressor.evaluate(input_fn = input_fn_eval,steps=1)
print(metrics_data)

#%%
import numpy as np
predicted = list(linear_regressor.predict(input_fn = input_fn_eval))
predicted = [p['predictions'][0] for p in predicted]
predicted = np.array(predicted)
actual = np.array(list(y_test))
# plt.plot(x_test['TAX'],predicted,'r.',label='Predicted')
# plt.plot(x_test['TAX'],y_test,'b.',label='Actual')
plt.plot(range(actual.size),(predicted-actual)**2,'b-',label='sqrd error')
plt.show()


#%%
