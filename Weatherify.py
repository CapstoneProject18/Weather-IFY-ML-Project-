
# coding: utf-8

# In[2]:


import pandas as pd
train = pd.read_csv("C:/Users/GAURAV.GAURAV-UG2-2118/Downloads/Compressed/hurricanes-and-typhoons-1851-2014/pacific.csv")
# calculate no of rows and columns
train_shape = train.shape
print(train_shape)


# In[4]:


#train["Maximum Wind"].value_counts()
#train["Status"].unique()
train["Maximum Wind"].unique()
#train["Minimum Pressure"].unique()


# In[10]:


train["Minimum Pressure"].value_counts()


# In[7]:


#train["Latitude"].value_counts()
train["Latitude"].unique()


# In[1]:


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


# In[22]:


columns = ['Maximum Wind']

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train[columns], train["Status_categories"])


# In[23]:


#Splitting our train data into two so as to predict the value and check for accuracy
from sklearn.model_selection import train_test_split

columns = ['Maximum Wind']
all_X = train[columns]
all_y = train['Status_categories']

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.20,random_state=0)


# In[29]:


#train_y.head()
#Making the predictions and measuring the accuracy
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)
accuracy = accuracy_score(test_y, predictions)
print(accuracy)


# In[32]:


#Doing cross validation and taking the mean of all the accuracy score of each fold 
from sklearn.model_selection import cross_val_score
import numpy as np
lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv=5)
accuracy = np.mean(scores)
print(scores)
print(accuracy)


# In[5]:


def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df
#Creating dummies for each unique value
train = create_dummies(train,"Maximum Wind")
train.head()


# In[6]:


train[22773:22780]


# In[7]:


#Assigning 70% data to train and 30% data to test
import numpy as np
shuffled_rows = np.random.permutation(train.index)
shuffled_train = train.iloc[shuffled_rows]
highest_train_row = int(train.shape[0] * .70)
train1 = shuffled_train.iloc[0:highest_train_row]
test = shuffled_train.iloc[highest_train_row:]


# In[15]:


#Checking train1 and test data
train1.shape
#test.shape


# In[12]:


#Making the model using the feature
from sklearn.linear_model import LogisticRegression

unique_status = train["Status"].unique()
unique_status.sort()

models = {}
features = [c for c in train1.columns if c.startswith("Maximum Wind")]

for status in unique_status:
    model = LogisticRegression()
    
    X_train = train1[features]
    y_train = train1["Status"] == status

    model.fit(X_train, y_train)
    models[status] = model


# In[13]:


testing_probs = pd.DataFrame(columns=unique_status)
for status in unique_status:
    # Select testing features.
    X_test = test[features]   
    # Compute probability of observation being in the status.
    testing_probs[status] = models[status].predict_proba(X_test)[:,1]


# In[14]:


#we use idxmax to return a Series where each value corresponds to the column or where the maximum value occurs for that observation
predicted_status = testing_probs.idxmax(axis=1)
print(predicted_status)


# In[16]:


train[features].head()

