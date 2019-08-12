import pandas as pd
filePath = "3-reading_files/sample_data_5.csv"


df1 = pd.read_csv(filePath)
print(df1)

df2 = df1.copy(deep=True)       #deep copy

#finding null values per column
print(df1.isnull().sum())

#filling null names
df1.loc[(df1["Name"].isnull())& (df1["Sex"]=="Male"),"Name"] = "DefMale"
df1.loc[(df1["Name"].isnull())& (df1["Sex"]=="Female"),"Name"] = "DefFemale"
print(df1)

#filling empty RegNo
df1.loc[df1["RegNo"].isnull(),"RegNo"] = 0
print(df1)

#filling empty age values with mean
mean_age = df1["Age"].mean(skipna=True)

df1.loc[df1["Age"].isnull(),"Age"] = mean_age
print(df1)

print()
print()

#we can also use .fillna() for above operations
df2["RegNo"] = df2["RegNo"].fillna(0)
print(df2)
df2["Age"] = df2["Age"].fillna(df2["Age"].mean())
print(df2)