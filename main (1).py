#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn


# In[2]:


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
output_ids=test["PassengerId"]
test.head(5)
def clean(x):
    x=x.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
    columns=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    for col in columns:
        x[col].fillna(x[col].mean(),inplace=True)
    return x
def change():
    from sklearn import preprocessing
    LE=preprocessing.LabelEncoder()
    cols=["Sex","Embarked"]
    for col in cols:
        train[col]=LE.fit_transform(train[col])
        test[col]=LE.transform(test[col])
change()
train=clean(train)
test=clean(test)


# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
y=train["Survived"]
x=train.drop("Survived",axis=1)
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=42)
from xgboost import XGBClassifier
model=XGBClassifier()
model.fit(x_train,y_train)
prediction=model.predict(x_val)
#titanic=LogisticRegression(random_state=0,max_iter=1000).fit(x_train,y_train)


# In[32]:


#prediction=titanic.predict(x_val)
from sklearn.metrics import accuracy_score
accuracy_score(y_val,prediction)


# In[27]:


output=model.predict(test)


# In[28]:


df=pd.DataFrame({"PassengerId":output_ids.values,"Survived":output})
df.to_csv("submission.csv",index=False)


# In[ ]:





# In[ ]:
