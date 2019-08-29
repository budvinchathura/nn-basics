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

file_path = "9-classification/weatherAUS.csv"
weather_AUS_df = pd.read_csv(file_path)
print(weather_AUS_df)
print()
print("null data:")
print(weather_AUS_df.isnull().sum())

#%%
column_names = list(weather_AUS_df.columns)
column_names.remove('Date')

text_columns = ['Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']
for i in range(len(text_columns)):
    column_names.remove(text_columns[i])



# plot for analysing

# for i in range(len(column_names)):
#     plt.hist(weather_AUS_df[column_names[i]])
#     plt.title(column_names[i])
#     plt.show()

# for i in range(len(text_columns)):
#     frequencies= weather_AUS_df[text_columns[i]].value_counts()
#     buckets = list(range(len(list(frequencies.index))))
#     plt.bar(buckets,frequencies.values)                 # plot our bars
#     plt.xticks(buckets,list(frequencies.index))         # add lables
#     plt.title(text_columns[i])
#     plt.show()


#%%
# drop unwanted
weather_AUS_df.drop(['RISK_MM','Date'],inplace=True,axis = 1)

# too many missing data
weather_AUS_df.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm'],inplace=True,axis = 1)

print("null data:")
print(weather_AUS_df.isnull().sum())



#%%
# fill missing data
all_columns = list(weather_AUS_df.columns)

categorical_columns = ['Location', 'WindGustDir',
                       'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
numerical_columns = all_columns[::]

for cat_col in categorical_columns:
    numerical_columns.remove(cat_col)

for num_col in numerical_columns:
    mean_val = weather_AUS_df[num_col].mean()
    weather_AUS_df[num_col].fillna(mean_val,inplace = True)

    # standardize
    mean_val = weather_AUS_df[num_col].mean()
    std_dev = weather_AUS_df[num_col].std()

    weather_AUS_df[num_col] = weather_AUS_df[num_col].apply(lambda x:(x-mean_val)/std_dev)

for cat_col in categorical_columns:
    # fill with most common value
    mode = list(weather_AUS_df[cat_col].mode(dropna = True))[0]
    weather_AUS_df[cat_col].fillna(mode,inplace = True)

# for classification
weather_AUS_df['RainTomorrow'] = weather_AUS_df['RainTomorrow'].map({'No':0,'Yes':1})

print("null data:")
print(weather_AUS_df.isnull().sum())

# print(weather_AUS_df)



#%%

features = weather_AUS_df.drop('RainTomorrow',axis=1)
target = weather_AUS_df['RainTomorrow']


x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=55)

#%%
# seperating feature columns
numeric_features = [tf.feature_column.numeric_column(key=column) for column in numerical_columns]

categorical_columns.remove('RainTomorrow')
categorical_features = [tf.feature_column.categorical_column_with_vocabulary_list(key=column,vocabulary_list=features[column].unique()) for column in categorical_columns]

linear_features = numeric_features + categorical_features

# print(features)

#%%
# use builtin input functions

# shuffle parameter is compulsory
input_fn_train = tf.estimator.inputs.pandas_input_fn(
    x=x_train, y=y_train, batch_size=MINIBATCH_SIZE, num_epochs=None,shuffle=True)

input_fn_eval = tf.estimator.inputs.pandas_input_fn(x = x_test,y=y_test,batch_size = MINIBATCH_SIZE,num_epochs = 1,shuffle = False)

print(features.dtypes)
print(target.dtypes)

#%%

# tf.reset_default_graph()


# instantiate and run model
# model_dir location to save the model

linear_classifier = tf.estimator.LinearClassifier(feature_columns = linear_features,model_dir = "9-classification/save_data_for_1-binary.py_")
linear_classifier.train(input_fn = input_fn_train,steps = NUM_STEPS)

#%%
# evaluate with test data
metrics_data = linear_classifier.evaluate(input_fn = input_fn_eval,steps=1)
print(metrics_data)

#%%

# predict and visualize

predicted = list(linear_classifier.predict(input_fn = input_fn_eval))
predicted = [p['classes'][0] for p in predicted]
predicted = np.array(predicted,dtype=int)
actual = np.array(list(y_test),dtype=int)
diff = pd.Series(predicted-actual)


frequencies= diff.value_counts()
plt.bar([0,1,2],[frequencies[-1],frequencies[0],frequencies[1]])                 # plot our bars
plt.xticks([0,1,2],['-1','0','+1'])         # add lables
plt.title('(predicted - actual) distribution')
plt.show()



#%%
