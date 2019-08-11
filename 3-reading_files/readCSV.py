# CSV stands for comma seperated values
import pandas as pd
filePath = "3-reading_files/sample_data.csv"

#indexes are added automatically
#starting from 0
df1 = pd.read_csv(filePath)
print(df1)          #prints whole dataframe
print(df1.head(n=3))   #returns first n data rows, default is 5
print(df1.index)
print(df1.columns)
print(df1.values)

# seperator can be a custom character
filePath2 = "3-reading_files/sample_data_2.csv"
df2 = pd.read_csv(filePath2,sep=";")            
print(df2.head())                       #this is a huge data dump, we only print head

#character seperated files can be also in .txt extension
filePath3 = "3-reading_files/sample_data_3.txt"
df3 = pd.read_csv(filePath3)
print(df3.head())