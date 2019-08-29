#%%
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import datasets, metrics, preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time

NUM_STEPS = 2000
MINIBATCH_SIZE = 16

#%%

# dataset from https://www.kaggle.com/kumarajarshi/life-expectancy-who

filePath = "8-estimators/life_expectancy.csv"
life_expec_df = pd.read_csv(filePath)
print(life_expec_df.head())
print()
print(life_expec_df.isnull().sum())

#%%
column_names = list(life_expec_df.columns)
column_names.remove('Country')
column_names.remove('Year')
column_names.remove('Status')





# plot as histogram for analysing

# for i in range(len(column_names)):

#     plt.hist(life_expec_df[column_names[i]])
#     plt.title(column_names[i])
#     plt.show()

#%%

# drop unwanted
life_expec_df.drop(['infant_deaths','percentage_expenditure','Measles','under-five_deaths','HIV_AIDS','Population'],inplace=True,axis=1)

# drop GDP and Hepatitis_B beacuse many rows are missing those data
life_expec_df.drop(['GDP','Hepatitis_B'],inplace=True,axis=1)

#%%
# fill null data

column_names = list(life_expec_df.columns)
column_names.remove('Country')
column_names.remove('Year')
column_names.remove('Status')

for column in column_names:
    mean_val = life_expec_df[column].mean()
    life_expec_df[column].fillna(mean_val,inplace=True)

print(life_expec_df.isnull().sum()) 

#%%
features = life_expec_df.drop('Life_expectancy', axis=1)
target = life_expec_df['Life_expectancy']

# 30% for testing
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=55)


#%%
# seperating feature columns

all_columns = list(features.columns)

numeric_columns = all_columns[::]
numeric_columns.remove('Country')
numeric_columns.remove('Year')
numeric_columns.remove('Status')

categorical_columns = ['Country','Year','Status']

# print(sorted(life_expec_df['Country'].unique()))
# print(sorted(life_expec_df['Year'].unique()))
# print(sorted(life_expec_df['Status'].unique()))

numeric_features = [tf.feature_column.numeric_column(
    key=column) for column in numeric_columns]

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

# tf.reset_default_graph()


# instantiate and run model
# model_dir location to save the model

linear_regressor = tf.estimator.LinearRegressor(feature_columns = linear_features,model_dir = "8-estimators/save_data_for_2.py")
linear_regressor.train(input_fn = input_fn_train,steps = NUM_STEPS)

#%%
# evaluate with test data
metrics_data = linear_regressor.evaluate(input_fn = input_fn_eval,steps=1)
print(metrics_data)

#%%

# visualize

predicted = list(linear_regressor.predict(input_fn = input_fn_eval))
predicted = [p['predictions'][0] for p in predicted]
predicted = np.array(predicted)
actual = np.array(list(y_test))

sqrd_errors = (predicted-actual)**2
plt.hist(sqrd_errors)
plt.title('distribution of sqrd error for test data')
plt.show()