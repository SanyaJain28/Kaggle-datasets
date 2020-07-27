#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import sklearn


# In[2]:


data=pd.read_csv('titanic_train.csv')
data.hist(bins=50, figsize=(20,15))


# In[3]:


data.info()


# In[4]:


data.describe(include='all')


# In[5]:


data['age'].isnull().sum()


# In[6]:


data['age'].isnull()


# In[7]:


sns.heatmap(data.isnull())#, yticklabels=False, cmap='BuPu')


# In[8]:


df=data.copy()
df


# In[9]:


sns.set_style("ticks")
sns.countplot(df['survived'])


# In[10]:


sns.set_style("dark")
sns.countplot(df['survived'], hue=df['sex'], palette='coolwarm_r' )


# In[11]:


df.groupby(['sex', 'survived'] )['survived'].count().plot.bar(figsize=(8, 6))


# In[12]:


df.groupby(['sex', 'survived'] )['survived'].count().unstack(1)


# In[13]:


sns.set_style("darkgrid")
sns.countplot(df['survived'], hue=df['pclass'], palette='rainbow' )


# In[14]:


sns.boxplot(df['pclass'], df['age'])


# In[15]:


sns.countplot(df['sibsp'])


# In[16]:


sns.countplot(df['parch'])


# In[17]:


a=df[['pclass', 'survived']].groupby(df['pclass'],as_index=False).mean().sort_values(by='survived', ascending =False)
a


# In[18]:


df[['sex','survived']].groupby(df['sex'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[19]:


pd.crosstab(df['sex'], df['survived'], margins=True)


# In[20]:


df.groupby(df['sex']).survived.mean()


# In[21]:


df[['pclass','survived']].groupby(df['pclass'], as_index=False).mean()


# In[22]:



table = pd.crosstab(df['survived'],df['pclass'])
table


# In[23]:


df['age'].plot.hist() # or use df['age'].plot(kind='hist')  or sns.distplot(df['age'].dropna(),kde=False, bins=20)


# In[24]:


df['age'].fillna(df.age.mean(), inplace=True) #or df['age'].fillna(29, inplace=True)
df


# In[25]:


df['fare'].plot.hist() or sns.distplot(df['fare'].dropna(),kde=False, bins=20)


# In[26]:


df.drop(['body'], axis=1, inplace=True)


# In[27]:


df.drop(['cabin'], axis=1, inplace=True)


# In[28]:


df.drop(['boat'], axis=1, inplace=True)


# In[29]:


df.drop(['home.dest'], axis=1, inplace=True)


# In[30]:


df.drop(['passenger_id'], axis=1, inplace=True)


# In[31]:


df.isnull().sum()


# In[32]:


df_nomissval=df.dropna()


# In[33]:


df_nomissval.describe()


# In[34]:


y=df_nomissval.corr()


# In[35]:


sns.heatmap(y)


# In[36]:


sns.pairplot(df_nomissval)


# In[37]:


sns.distplot(df_nomissval['age'])


# In[38]:


sns.distplot(df_nomissval['fare']) # we can see some outliers and so we should remove them


# In[39]:


df_nomissval.drop(['ticket'], axis=1, inplace=True)


# In[40]:


df_nomissval


# In[41]:


q=df_nomissval['fare'].quantile(.05)
df_fareout=df_nomissval[df_nomissval['fare']>q]
sns.distplot(df_nomissval['fare'])


# In[42]:


df_fareout.describe(include='all')


# In[43]:


df_nomissval.describe(include='all')


# In[44]:


df_mapped=df_nomissval.copy()


# In[45]:


df_mapped['sex']=df_mapped['sex'].map({'male':0, 'female':1})


# In[46]:


#df_mapped['embarked']=df_mapped['embarked'].map({'Q':0, 'S':1, 'C':2}) didnt work
df_mapped['embarked'] = df_mapped['embarked'].map( {'S': 1, 'C': 2, 'Q': 3} )
df_mapped


# In[47]:


df_mapped.describe(include='all')


# In[48]:


targets=df_mapped['survived']
df_mapped.shape


# In[49]:


df_mapp_new=df_mapped.drop(['name'], axis=1, inplace=True)


# In[50]:


unscaled_data=df_mapped.iloc[:, :7]
unscaled_data


# In[51]:


from sklearn.preprocessing import StandardScaler


# In[52]:


survived=StandardScaler()


# In[53]:


survived.fit(unscaled_data)


# In[54]:


scaled_data=survived.transform(unscaled_data)


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


train_test_split(scaled_data, targets)


# In[57]:


x_train, x_test, y_train, y_test=train_test_split(scaled_data,targets, train_size=0.8, random_state=1)


# In[58]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[59]:


reg=LogisticRegression()


# In[60]:


reg.fit(x_train, y_train)


# In[61]:


predictions=reg.predict(x_train)


# In[62]:


np.sum(predictions==y_train)


# In[63]:


x_train.shape


# In[64]:


y_train.shape


# In[65]:


reg.score(x_train, y_train)


# In[66]:


from sklearn.metrics import classification_report # only used for classification models


# In[67]:


print(classification_report(y_train,predictions))


# In[68]:


from sklearn.metrics import confusion_matrix


# In[69]:


print(confusion_matrix(y_train,predictions))


# In[77]:


ypredictions=reg.predict(x_test)


# In[75]:


reg.score(x_test, y_test)


# In[78]:


print(confusion_matrix(y_test,ypredictions))


# In[79]:


print(classification_report(y_test,ypredictions))

