#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly_express as px
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv(r"C:\Users\Muskan Khan\OneDrive\Documents\JUPYTER\iPhone_Sales_Prediction\Order_details(masked).csv")
print(data)


# In[3]:


null_col=data.isnull().all()
print(null_col)


# In[4]:


print(data.shape)


# In[5]:


data.head()


# In[6]:


data['Time']=pd.to_datetime(data['Transaction Date'])
data['Hour']=(data['Time']).dt.hour


# In[7]:


data.head()


# In[8]:


#Note: counts shows how many varaibles are there in the columns whilts the value shows the variables frequency!!


# In[9]:


# Calculate the frequency of each hour and sort by hour
timemost=data['Hour'].value_counts().sort_index()

# Ensure all hours from 0 to 23 are included, fill missing hours with 0
timemost=timemost.reindex(range(24), fill_value=0)


# In[10]:


# time1=data['Hour'].value_counts().index.tolist()[:24]
# time2=data['Hour'].value_counts().values.tolist()[:24]


# Extract hours and counts
time1=timemost.index.tolist()
time2=timemost.values.tolist()


# In[11]:


hourly_order_df=pd.DataFrame({
    'Hour1': time1,
    'Order_Count': time2
})

print(hourly_order_df)


# In[12]:


fig=px.line(hourly_order_df, x='Hour1', y='Order_Count',
           title='Sales Happening Per Hour (Spread Throughout The Week)',
            labels={'Order_Counnt': 'Number Of Purchases Made'},
            line_shape='linear')

fig.update_layout(
    xaxis_title='Hour',
    yaxis_title='Number Of Purchases Made',
    title_font=dict(size=20),
    xaxis_title_font=dict(size=14),
    yaxis_title_font=dict(size=14)
)

fig.show()


# In[13]:


# By Matplot

plt.figure(figsize=(12,6))

plt.plot(time1, time2, color='m')
plt.title('Sales Happening Per Hour (Spread Throughout The Week)', fontsize=20)
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Number Of Purchases Made', fontsize=14)
plt.grid(True)
plt.show()
           


# # Concating Data

# Not necessary here But for practice use! 

# In[14]:


# Merging with main data

# frames=[data, hourly_order_df]
# concat_data=pd.concat(frames)..... is wali method se NaN values aa rhi thi idk why!


concat_data=pd.concat([data, hourly_order_df], axis=1)
concat_data.head()


# # Training & Testing 

# In[15]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[16]:


x=concat_data[['Hour1']]
y=concat_data['Order_Count']

Xtrain,Xtest, Ytrain, Ytest=train_test_split(x,y, test_size=0.10, random_state=42)


# In[17]:


print(concat_data[['Hour1', 'Order_Count']].isna().sum)


# # Creating Pipeline

# In[18]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipeline= Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

XtrainNew=pipeline.fit_transform(Xtrain)
XtestNew=pipeline.transform(Xtest)

Yimputer=SimpleImputer(strategy='mean')
YtrainNew=Yimputer.fit_transform(Ytrain.values.reshape(-1,1)).ravel()
YtestNew=Yimputer.transform(Ytest.values.reshape(-1,1)).ravel()


# # Model Preparation and Prediction

# In[19]:


model=RandomForestRegressor()
model.fit(XtrainNew, YtrainNew)


# In[20]:


# some_xtest_data=XtestNew[:5]
# some_ytest_data=YtestNew[:5]
# prediction=model.predict(some_xtest_data)

prediction=model.predict(XtestNew[:5])
print(prediction)      #sales prediction per hour!


# In[21]:


print(XtestNew[:5].shape)
print(prediction.shape)


# In[22]:


# got error so checked the shape and then use ravel to flattened the array


# In[23]:


compare= pd.DataFrame({'Actual':XtestNew[:5].ravel(),
                       "Predicted": prediction.ravel()})
print(compare)


# # Checking for Accuracy

# In[24]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

my_prediction_model=model.predict(XtrainNew)

mae=mean_absolute_error(YtrainNew,my_prediction_model)
mse=mean_squared_error(YtrainNew, my_prediction_model)
rmse=np.sqrt(mse)

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:',rmse)


# In[25]:


whole_test_pred=model.predict(XtestNew)

comaprison_whole_test_data=pd.DataFrame({
    'Actual':YtestNew,
    'Predicted': whole_test_pred
})

print(comaprison_whole_test_data)


# In[26]:


test_msa=mean_absolute_error(YtestNew,whole_test_pred)
test_mse=mean_squared_error(YtestNew, whole_test_pred)
test_rmsa=np.sqrt(test_msa)

print(test_msa)
print(test_mse)
print(test_rmsa)


# # Cross Validation

# In[27]:


from sklearn.model_selection import cross_val_score

score=cross_val_score(model, XtrainNew, YtrainNew)
rmse_score=np.sqrt(score)

print(rmse_score)


# In[28]:


YtrainNew.min()


# In[29]:


YtrainNew.max()


# # Saving Model for Future Use

# In[30]:


from joblib import load, dump
dump(model,'iPhone_Sales_Prediction' )


# In[31]:


model=load('iPhone_Sales_Prediction')
features=np.array([[7]])
print(model.predict(features))


# # Using The Model In Prediction
# 

# In[32]:


features=np.array([[7], [8],[9]])
print(model.predict(features))

