
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# In[2]:


df= pd.read_csv('employee_data.csv')
#df.head()

############################### EDA ##############################
"""
df.info()

df['status'].value_counts()
sns.countplot(x='status',data=df)
# plottinig of Na values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

# calculating null values for all the columns
df.apply(lambda x:x.isnull().sum())

"""
######################### DATA IMPUTATION ###############################

#converting Na's to 0 for columns 'filed_complaint' & 'recently_promoted'
df['filed_complaint'].replace(np.nan,0,inplace=True)
df['recently_promoted'].replace(np.nan,0,inplace = True)

#imputing na with mean value for column 'Last_evaluation' & 'satisfaction'
df['last_evaluation'].fillna((df['last_evaluation'].mean()),inplace=True)
df['satisfaction'].fillna((df['satisfaction'].mean()),inplace=True)

# replacing na's with mode in tenure column
df['tenure'].fillna((df['tenure'].mode()[0]),inplace=True)
df['department'].fillna((df['department'].mode()[0]),inplace=True)

#converting 'status column into binary'
df['status'].replace(('Left','Employed'),(0,1),inplace=True)


# In[7]:

"""
plt.figure(figsize=(12,6))
sns.countplot(x='department',data=df,hue='tenure')
plt.legend(loc=1)


sns.distplot(df['n_projects'])


plt.figure(figsize=(15,6))
sns.countplot(x='department',data=df,hue='status')
plt.tight_layout()

plt.figure(figsize=(15,6))
sns.countplot(x='department',data=df,hue='n_projects')
plt.tight_layout()


sns.countplot(x='salary',data=df,hue='status')

sns.boxplot(x='n_projects',y='avg_monthly_hrs',data=df)

# plot for salary vs department
plt.figure(figsize=(15,6))
sns.countplot(x='department',data = df,hue='salary')

#plot for tenurewise salary
plt.figure(figsize=(10,5))
sns.countplot(x='salary',data=df,hue='tenure')


# plot deparment vs status
df.hist(column="avg_monthly_hrs",by="status",bins=30)


#creating bins for avg monthly hrs
bins = [45, 100, 150, 200, 250, 320]
df['binned_hrs'] = pd.cut(df['avg_monthly_hrs'], bins)


plt.figure(figsize=(15,8))
sns.countplot(x='department',data=df,hue='binned_hrs')

plt.legend(loc=1)


# creating subset of Data frame and creating pivot table out of that subset
df1=df[df['department']=='sales']
impute_grps = df1.pivot_table(values=["satisfaction"], index=['status',"salary","department"], aggfunc=np.mean)
print(impute_grps)

#creating bins for satisfaction
bins = [0, 0.25,0.50,0.75, 1]
df['binned_satisaction'] = pd.cut(df['satisfaction'], bins)

df['binned_satisaction'].value_counts()
#df['binned_hrs'].value_counts()


sns.countplot(x='binned_hrs',data=df,hue='status')



sns.countplot(x='binned_satisaction',data=df,hue='status')



sns.heatmap(df.corr(),annot=True,cmap='coolwarm')



table = df.pivot_table(values=["status"], index=['binned_satisaction','binned_hrs'], aggfunc=np.sum)
print(table)

#employee with low satisfaction and more working hours leaves company

"""

#################### Creating Model ##################
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# converting salary column values into 1-3
df['salary'].replace(('low','medium','high'),(1,2,3),inplace=True)

#applying one hot encoding on department column by creating dummies
dummies=pd.get_dummies(df['department'])
df[['IT', 'admin', 'engineering', 'finance', 'information_technology',
       'management', 'marketing', 'procurement', 'product', 'sales', 'support',
       'temp']]=dummies
#drop columns department,recently_promoted,filed_complaint
df.drop(['department','recently_promoted','filed_complaint'],axis=1,inplace=True)


#creating X & y variable for train test split
X=df.drop(['status'],axis=1) #'binned_hrs','binned_satisaction'
y=df['status']


# In[40]:


#splitting Data into train,test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=100)

# In[41]:


#Random forest classifier model
rfc=RandomForestClassifier(n_estimators=100)


# In[42]:


rfc.fit(X,y)
from sklearn.externals import joblib
joblib.dump(rfc, 'model.pkl')
# In[43]:


# prediciting for test dataset
#rfc_pred = rfc.predict(X_test)





# In[44]:
"""

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[45]:


# evaluating model's goodness of fit
print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(accuracy_score(y_test,rfc_pred))


# In[46]:


print(classification_report(y_test,rfc_pred))


# In[48]:


# logistic regression
from sklearn.linear_model import LogisticRegression


# In[49]:


logmodel=LogisticRegression()


# In[50]:


logmodel.fit(X_train,y_train)


# In[51]:


pred=logmodel.predict(X_test)


# In[54]:



print(confusion_matrix(y_test,pred))
print('\n')
print(accuracy_score(y_test,rfc_pred))


# In[55]:


######################## DECISION TREE ###################

from sklearn.tree import DecisionTreeClassifier


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=100)
treemodel=DecisionTreeClassifier()
treemodel.fit(X_train,y_train)
pred=treemodel.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(accuracy_score(y_test,pred))


# In[56]:


################## RAW DATA PREDICTIONS #############################

test_data=pd.read_csv('unseen_raw_data.csv')
test_data.head()


# In[57]:


test_data.apply(lambda x:x.isnull().sum())


# In[58]:


#converting Na's to 0 for columns 'filed_complaint' & 'recently_promoted'
test_data['filed_complaint'].replace(np.nan,0,inplace=True)
test_data['recently_promoted'].replace(np.nan,0,inplace = True)

#imputing na with mean value for column 'Last_evaluation' & 'satisfaction'
test_data['last_evaluation'].fillna((test_data['last_evaluation'].mean()),inplace=True)
test_data['satisfaction'].fillna((test_data['satisfaction'].mean()),inplace=True)

# replacing na's with mode in tenure column
test_data['tenure'].fillna((test_data['tenure'].mode()[0]),inplace=True)
test_data['department'].fillna((test_data['department'].mode()[0]),inplace=True)


# In[59]:


test_data['salary'].replace(('low','medium','high'),(1,2,3),inplace=True)


# In[62]:


dummies=pd.get_dummies(test_data['department'])
test_data[['IT', 'admin', 'engineering', 'finance', 'information_technology',
       'management', 'marketing', 'procurement', 'product', 'sales', 'support',
       'temp']]=dummies
test_data.drop(['department','recently_promoted','filed_complaint'],axis=1,inplace=True)


# In[63]:


#fitting random forest model on whole train dataset

rfc.fit(X,y)


# In[64]:


#predicting for test dataset
prediction_unseendata=rfc.predict(test_data)


# In[65]:


# inserting predicted value in test dataset
test_data['status']=prediction_unseendata


# In[66]:


test_data.head()


# In[388]:


#creating test dataset file with predicted value
#test_data.to_csv('predictedfile.csv')


# In[67]:


predict_file=pd.read_csv('unseen_raw_data.csv')


# In[68]:


# INSERTING PREDICTIONS VALUE ON RAW TEST DATASET
predict_file['status']=prediction_unseendata


# In[69]:


# Saving dataset.
predict_file.to_csv('FINALPREDICTIONFILE.csv')

"""
