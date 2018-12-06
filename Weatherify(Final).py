
# coding: utf-8

# In[1]:


import pandas as pd
train = pd.read_csv("C:/Users/GAURAV.GAURAV-UG2-2118/Downloads/Compressed/hurricanes-and-typhoons-1851-2014/pacific.csv")
# calculate no of rows and columns
train_shape = train.shape
print(train_shape)


# In[2]:


def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df
#Creating dummies for each unique value
train = create_dummies(train,"Maximum Wind")
train = train.drop("Maximum Wind",axis=1)
train = create_dummies(train,"Minimum Pressure")
train = train.drop("Minimum Pressure",axis=1)
train = create_dummies(train,"Date")
train = train.drop("Date",axis=1)
train.head()


# In[3]:


#Assigning 70% data to train1 and 30% data to test
import numpy as np
shuffled_rows = np.random.permutation(train.index)
shuffled_train = train.iloc[shuffled_rows]
highest_train_row = int(train.shape[0] * .70)
train1 = shuffled_train.iloc[0:highest_train_row]
test = shuffled_train.iloc[highest_train_row:]


# In[4]:


train1.shape
test.shape


# In[5]:


#Making the model using the features
from sklearn.linear_model import LogisticRegression

unique_status = train["Status"].unique()
unique_status.sort()

models = {}
features = [c for c in train1.columns if c.startswith("Maximum Wind") or c.startswith("Minimum Pressure") or c.startswith("Date")]

for status in unique_status:
    model = LogisticRegression()
    
    X_train = train1[features]
    y_train = train1["Status"] == status

    model.fit(X_train, y_train)
    models[status] = model


# In[7]:


testing_probs = pd.DataFrame(columns=unique_status)
for status in unique_status:
    # Select testing features.
    X_test = test[features]   
    # Compute probability of observation being in the status.
    testing_probs[status] = models[status].predict_proba(X_test)[:,1]


# In[8]:


#we use idxmax to return a Series where each value corresponds to the column or where the maximum value occurs 
#for that observation
predicted_status = testing_probs.idxmax(axis=1)
print(predicted_status)


# In[9]:


#Measuring the accuracy of the model
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test["Status"], predicted_status)
print(accuracy)


# In[10]:


#Finding the cross validation of the model i.e evaluating our model
from sklearn.model_selection import cross_val_score
cross_val = cross_val_score(LogisticRegression(), train[features], train["Status"], scoring='accuracy', cv=10)
print (cross_val)
print (cross_val.mean())


# In[11]:


# Finding the confusion matrix to see the no of correct instances
from sklearn import metrics
print (metrics.confusion_matrix(test["Status"], predicted_status))


# In[ ]:


# Doing feature selection
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
# Build RF classifier to use in feature selection
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Build step forward feature selection
sfs1 = sfs(clf,
           k_features=3,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=0)

# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)

# Naming the final features selected
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)

