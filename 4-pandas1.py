import pandas as pd
import numpy as np

dict1 = {"a": 3, "b": 7, "c": None}
series1 = pd.Series(dict1)
print(series1)

# first_param: data    second_param:indexes

# one dimension
oneD = pd.Series([45, 7, 9, 0, -5], ["a", "b", "c", "d", "e"])
print(oneD)

# or
anotherOneD = pd.Series(data=[1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
print(anotherOneD)

print(oneD.loc[["a", "b"]])  # specific values

# or using 0 based index, filtered by the order of indexes
print(oneD[[0, 3, 4]])

# another method
# only 1 index is allowed
print(oneD.iloc[0])  # returns value

print("a" in oneD)  # true, key exists
print("B" in oneD)  # false, no key as 'B'


# creating a dataframe
data = {
    "A": pd.Series([100, 200, 350], ["SL", "ENG", "WI"]),
    "B": pd.Series([45.6, 33.8, 22.1, 23.7], ["ENG", "SL", "IND", "WI"]),
}

df = pd.DataFrame(data)
print("\n")
print(df)  # notice: 'IND' assigned NaN for column 'A'

# another method
# data should be passed in row wise array format
myData = np.array([[100, 200, 350, 0], [45.6, 33.8, 22.1, 23.7]])
df2 = pd.DataFrame(
    # using transpose becuase row wise format is needed
    data=myData.transpose(), index=["ENG", "SL", "IND", "WI"], columns=["A", "B"]
)

print()
print(df2)

print(df2.index)            #prints ["ENG", "SL", "IND", "WI"]
print(df2.values)           #prints row wise values
print(df2.columns)          #prints ['A','B']

# filtering dataFrame
filtered = pd.DataFrame(df2,index=['SL','IND'],columns=['A'])
print(filtered)

total = df + df2
print(total)            #perform addition relevant values

