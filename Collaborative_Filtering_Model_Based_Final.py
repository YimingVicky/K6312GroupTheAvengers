#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Collaborative Filtering - user based


# In[2]:


# Similar users can used to predict how much a user like a product never used before


# In[3]:


# user-based filtering


# In[4]:


# use Surprise library with SVD and min RMSE 


# In[5]:


# https://surprise.readthedocs.io/en/stable/getting_started.html
# https://surprise.readthedocs.io/en/stable/FAQ.html#raw-inner-note


# In[6]:


from surprise import Reader
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise import accuracy


# In[7]:


import pandas as pd
import numpy as np

# import datasets - 100k small
ML_ratings = pd.read_csv('ratings.csv')


# In[8]:


reader = Reader()


# In[10]:


# Load the movielens-100k dataset 
data = Dataset.load_from_df(ML_ratings[['userId', 'movieId', 'rating']], reader)


# In[11]:


svd = SVD()   # using SVD algorithm

# In[13]:


# Retrieve the trainset - train on full set
trainset = data.build_full_trainset()  
# Build an algorithm, and train it.
svd.fit(trainset)














