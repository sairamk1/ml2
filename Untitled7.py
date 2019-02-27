#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)
boston.keys()
print (boston.feature_names)
print (boston.DESCR)


# In[35]:


bos.head()


# In[36]:


bos.columns = boston.feature_names
bos.head()


# In[37]:


boston.target[:5]


# In[48]:


bos['price'] = boston.target


# In[51]:


plt.hist(boston.target,bins=20)
plt.suptitle('Boston Housing Prices in $1000s', fontsize=15)
plt.xlabel('Prices in $1000s')
plt.ylabel('Count')
plt.show()


# In[52]:


df = pd.concat([pd.DataFrame(boston.data, columns=boston.feature_names), pd.DataFrame(boston.target, columns=['MEDV'])], axis=1)
df.describe()


# In[56]:


#OK let's now fit a LinearRegression model to this dataset. We'll start with a very simple classifier in LinearRegression.
#This uses the least squares method I'd mentioned earlier.
#To make it easier for us to visualize this dataset (and how our model fits), let's use PCA to reduce this to a
#single dimension. For more information about PCA, refer to a seperate notebook on PCA in my repo.
data_reduced=PCA(n_components=1).fit_transform(boston.data) 
#Let's now split the dataset into train and test sets so we can find out how well the model can generalize.
X_train, X_test, y_train, y_test = train_test_split(data_reduced, boston.target)
print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Let's fit the LinearRegression classifier to the training set 
linr=LinearRegression().fit(X_train, y_train) 
#We'll now run predictions on the Test set using the model we just trained 
y_pred = linr.predict(X_test) 
#Let's check out the score - in this case, this is the R-squared which tells us how much of the  
#variance of the data is captured by the model. The higher this number is, the better. 
print ("R-squared for train: %.2f" %linr.score(X_train, y_train))
print ("R-squared for test: %.2f" %linr.score(X_test, y_test))
#That's pretty reasonable. We're able to capture about 67% of variance in the test dataset.- See more at: https://shankarmsy.github.io/stories/linear-reg-sklearn.html#sthash.gbquIxEZ.dpuf


# In[ ]:




