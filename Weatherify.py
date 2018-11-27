
# coding: utf-8

# In[2]:


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


# In[19]:


#Converting Text Column into Numeric Column
from sklearn.preprocessing import LabelEncoder
df = train["Status"]

le = LabelEncoder()
le.fit(df)
list(le.classes_)
train["Status_categories"]=le.transform(df) 
#train.head(60)


# In[21]:


#Visualising the feature column
pivot = train.pivot_table(index="Maximum Wind",values='Status_categories')
pivot.plot.bar()
plt.show()

pivot1 = train.pivot_table(index="Minimum Pressure",values='Status_categories')
pivot1.plot.bar()
plt.show()

