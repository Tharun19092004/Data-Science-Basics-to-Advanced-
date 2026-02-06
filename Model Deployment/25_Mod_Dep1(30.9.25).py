#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pickle


# In[2]:


log_model=pickle.load(open('Model Deployment/log30.pkl','rb'))


# In[3]:


st.title('Model Deployment using Logistic Regression')


# In[4]:


def user_input_parameter():
    Gender= st.sidebar.selectbox('Select your Gender ,Male-1,Female-0',[0,1])
    Insur= st.sidebar.selectbox('Select Insurance details, Yes-1, No-0',[0,1])
    seat= st.sidebar.selectbox('Select seatbelt details , Yes-1,No-0',[0,1])
    Age= st.sidebar.slider('Select your Age',0,100)
    loss= st.sidebar.number_input('Enter the loss')
    dict1= {'CLMSEX':Gender,'CLMINSUR':Insur,'SEATBELT':seat,'CLMAGE':Age,'LOSS':loss}
    features= pd.DataFrame(dict1, index=[0])
    return features
df= user_input_parameter()
pred=log_model.predict(df)
pred_prob= log_model.predict_proba(df)
button= st.button('Predict')
if button is True:
    st.subheader('Predicted')
    st.write('Eligible' if pred_prob[0][1]>=0.5 else 'Not Eligible')
    st.subheader('Pred_Prob')
    st.write(pred_prob)


# In[ ]:




