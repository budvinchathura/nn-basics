import pandas as pd

filePath = "3-reading_files/sample_data_4.xlsx"
xFile = pd.ExcelFile(filePath)
print(xFile.sheet_names)

df1 = xFile.parse(sheet_name="Sheet1")
print(df1.head())

print()

df2 = xFile.parse(sheet_name="Sheet2")
print(df2.head())