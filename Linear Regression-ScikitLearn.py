# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 03:34:10 2024

@author: harpr
"""
#%%
import pandas as pd
#%%
import numpy as np
#%%
import seaborn as sns
#%%
import matplotlib.pyplot as plt
#%%
from sklearn.linear_model import LinearRegression
#%%
np.random.seed(42)
num_samples =500
years_of_experience = np.random.randint(2,21, size=num_samples)
slope =(200_000 - 60_000)/18
intercept = 60_000
salaries = slope*years_of_experience+intercept+np.random.normal(0,10000)
data ={'Years_of_Experience': years_of_experience,'Salary':salaries}
df=pd.DataFrame(data)
#%%
plt.figure(figsize=(10,6))
sns.scatterplot(x='Years_of_Experience', y='Salary',data=df,color='blue')
sns.regplot(x='Years_of_Experience', y='Salary',data=df, scatter=False,color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression Salary')
plt.show()
#%%
X=df[['Years_of_Experience']]
y=df[['Salary']]
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=11)
#%%
lr=LinearRegression()
lr.fit(X_train, y_train)
#%%
lr.score(X_train, y_train)
#%%
lr.score(X_test, y_test)
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#%%
y_pred=lr.predict(X_test)
#%%
#%%


