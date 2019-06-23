#!/usr/bin/env python
# coding: utf-8

# In[58]:


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[59]:


iris=load_iris()      #loading iris data


# In[60]:


dir(iris)


# In[61]:


iris.feature_names


# In[62]:


iris.target_names


# In[63]:


features=iris.data


# In[64]:


label=iris.target
label.shape


# In[68]:


sl=features[0:,0]


# In[69]:


sw=features[0:,1]


# In[70]:


plt.xlabel("length")
plt.ylabel("width")
plt.scatter(sl,sw,label="sepal_data",marker='*')
plt.scatter(features[0:,2],features[0:,3],label="petal_data",marker='x')
plt.legend()


# In[71]:


#now time for data separating into two category
#training data
#testing data ----questions
from sklearn.model_selection import train_test_split
train_data,test_data,label_train,label_test=train_test_split(features,label,test_size=0.1)


# In[72]:


clf=DecisionTreeClassifier()


# In[73]:


trained=clf.fit(train_data,label_train)


# In[80]:


predicted_flowers=trained.predict(test_data)


# In[75]:


predicted_flowers


# In[76]:


label_test


# In[78]:


#accuracy
accuracy_score(predicted_flowers,label_test)


# In[ ]:





# In[ ]:




