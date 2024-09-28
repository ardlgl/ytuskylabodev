
import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv('Housing.csv')

df

df.head()

df.info()

df.isnull().sum()

df.describe().transpose()

print(df['mainroad'].value_counts())
print(df['guestroom'].value_counts())

import matplotlib.pyplot as plt

plt.hist(df['price'], bins=20, edgecolor='black')
plt.title('Distribution of Housing Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

df_numeric = pd.get_dummies(df, drop_first=True)

correlation_matrix = df_numeric.corr()

import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

X = df.drop(['price'], axis=1)

for categ in X.select_dtypes(include=['object']).columns:
    for num, col in enumerate(X.select_dtypes(exclude=['object']).columns):
        sns.catplot(x=categ, y=col, data=X, kind='point')
        plt.show()
numerical_columns = df.select_dtypes(include=[np.number]).columns
sns.pairplot(df[numerical_columns])
plt.show()

