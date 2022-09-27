#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


df=pd.read_csv('C:/Users/RIG1/Desktop/DS ASSIGNMENTS/QUESTIONS -all assignments/ASS 10/book.csv',encoding='latin-1')


# # While uploading the file I was getting following error:----------->
# 
# ### UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 26: invalid continuation byte
# 
# # That's why I wrote --------------------------> encoding='latin-1' 
# ### at the end of the line

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df1=df.iloc[:,1:]


# In[6]:


df1.head()
# OR df[0:5]


# # Renaming the column name

# In[7]:


df1.rename(columns = {'User.ID':'uid','Book.Title':'b_title','Book.Rating':'b_rating'}, inplace = True)


# # checking number of unique userid,book_title in dataset

# ### I was geting some column-name-error while checking----->len(df1.userid.unique()) so, thats why I renamed the columns.

# In[8]:


len(df1.uid.unique())


# In[9]:


len(df1.b_title.unique())


# In[10]:


df1.head()


# In[11]:


#df1.duplicated().sum()


# In[12]:


#df1[df1.duplicated()].shape


# In[13]:


#df1[df1.duplicated()]


# In[14]:


#cleandf=df1.drop_duplicates()


# In[15]:


#cleandf.shape


# In[16]:


#cleandf.duplicated().sum()


# In[17]:


clean_df2=df1.pivot_table(index='uid',columns='b_title',values='b_rating',aggfunc='sum')###.reset_index(drop=True)


# In[18]:


clean_df2.head()


# In[19]:


clean_df2.index=df1.uid.unique()


# In[20]:


clean_df2


# In[21]:


clean_df2.fillna(0,inplace=True)


# In[22]:


clean_df2


# In[23]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


# In[24]:


user_sim=1-pairwise_distances(clean_df2.values,metric='cosine')


# In[25]:


user_sim


# In[26]:


user_sim_df=pd.DataFrame(user_sim)


# In[27]:


user_sim_df.index=df1.uid.unique()
user_sim_df.columns=df1.uid.unique()


# In[28]:


user_sim_df#.iloc[0:10,0:10]


# In[29]:


np.fill_diagonal(user_sim,0)
user_sim_df#.iloc[0:10,0:10]


# # checking the higher similar customer

# In[30]:


user_sim_df.idxmax(axis=1)[0:10]


# In[31]:


df1[(df1['uid']==276729)|(df1['uid']==276726)]


# In[ ]:




