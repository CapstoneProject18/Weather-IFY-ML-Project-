
# coding: utf-8

# In[3]:


import pandas as pd
train = pd.read_csv("C:/Users/GAURAV.GAURAV-UG2-2118/Downloads/Compressed/hurricanes-and-typhoons-1851-2014/pacific.csv")
# calculate no of rows and columns
train_shape = train.shape
print(train_shape)


# In[8]:


train["Maximum Wind"].value_counts()


# In[10]:


train["Minimum Pressure"].value_counts()


# In[11]:


train["Latitude"].value_counts()

