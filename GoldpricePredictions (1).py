#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle


# Data collection and Processing

# In[6]:


#loading csv data to pandas dataframe
gold_data=pd.read_csv('gld_price_data.csv')


# In[7]:


#print first 5 rows in the data frame
gold_data.head()


# In[8]:


# print last 5 rows of the dataframe
gold_data.tail()


# In[9]:


# number of rows and columns
gold_data.shape


# In[10]:


# getting some basic informations about the data
gold_data.info()


# In[11]:


# checking the number of missing values
gold_data.isnull().sum()


# In[12]:


# getting the statistical measures of the data
gold_data.describe()


# 

# Correlation:
# 1. Positive Correlation
# 2. Negative Correlation

# In[13]:


correlation = gold_data.corr()


# In[15]:


# constructing a heatmap to understand the correlatiom
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Oranges')


# In[16]:


# correlation values of GLD
print(correlation['GLD'])


# In[17]:


# checking the distribution of the GLD Price
sns.displot(gold_data['GLD'],color='orange')


# 

# Splitting the Feature and Target

# In[18]:


X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']


# In[19]:


X


# In[20]:


Y


# Splitting into Training data and Test data

# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)


# Model training : Random Forest Regressor -Collection of decision Tress

# In[22]:


regressor = RandomForestRegressor(n_estimators=100)


# In[23]:


# training the model
regressor.fit(X_train,Y_train)


# Model Evaluation

# In[24]:


# prediction on Test Data
test_data_prediction = regressor.predict(X_test)


# In[25]:


test_data_prediction


# In[26]:


# R sqayre error
error_score=metrics.r2_score(Y_test,test_data_prediction)
print("R squared errror:",error_score)


# Comapre the Actual values and Predicted Values in a plot

# In[27]:


Y_test = list(Y_test)


# In[28]:


plt.plot(Y_test, color='maroon', label = 'Actual Value')
plt.plot(test_data_prediction, color='orange', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[31]:


#PICKEL FILE
pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# In[ ]:




