import pandas as pd 

df = pd.read_excel("data/Gen_AI Dataset.xlsx")

print(df['Query'].unique())
print(len(df['Query'].unique()))
print(df['Assessment_url'].unique())
print(len(df['Assessment_url'].unique()))

print(df.isnull().sum())