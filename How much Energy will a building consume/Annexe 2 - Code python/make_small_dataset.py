#!/usr/bin/env python
# coding: utf-8

# ## Import du jeu de données

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


train_raw = pd.read_csv('data/train.csv')


# In[3]:


building_meta_raw = pd.read_csv('data/building_metadata.csv')


# In[4]:


weather_raw = pd.read_csv('data/weather_train.csv')


# ## Sélection d'un échantillon pour tester

# ##### On sélectionne aléatoirement 100 building_id

# In[5]:


distinct_building_id = train_raw.building_id.unique()


# In[6]:


sample_building_id = np.random.RandomState(42).choice(distinct_building_id, 100, replace=False)


# ##### On filtre les trois tables en se basant sur cete échantillon

# In[7]:


train_sample = train_raw[train_raw.building_id.isin(sample_building_id)]


# In[8]:


building_sample = building_meta_raw[building_meta_raw.building_id.isin(sample_building_id)]


# In[9]:


weather_sample = weather_raw[weather_raw.site_id.isin(building_sample.site_id.unique())]


# ### Ecriture des jeux de données sample en csv

# In[10]:


train_sample.to_csv('data/train_sample.csv', sep=',', index=False)


# In[11]:


building_sample.to_csv('data/building_sample.csv', sep=',', index=False)


# In[12]:


weather_sample.to_csv('data/weather_sample.csv', sep=',', index=False)


# ### Jointure avec pandas

# In[13]:


m1 = pd.merge(building_sample, weather_sample, on='site_id', how='left')


# In[14]:


m2 = pd.merge(m1, train_sample, on=['building_id', 'timestamp'], how='right')


# In[15]:


df_final = m2.dropna(subset=['site_id'])


# In[16]:


df_final.to_csv('data/df_final.csv', index=False)


# In[18]:


df_final.info()


# ### Exploration

# ### Tri à plat

# In[17]:


for el in df_final.columns :
    print(el, len(df_final[df_final[el].isnull()]) / len(df_final))


# In[27]:


types = ['cat', 'cat', 'cat', 'int', 'int', 'int', 'ts', 'int', 'int', 'int', 'int', 'int', 'cat', 'int', 'int', 'int']
col = df_final.columns


# In[35]:


df_final


# In[16]:


building_meta_raw.info()


# In[32]:


vc = df_final.site_id.value_counts()


# In[38]:





# In[37]:


vc.values


# In[33]:


vc.index


# In[42]:


df_final.boxplot('air_temperature')


# In[41]:


plt.boxplot(df_final['air_temperature'])


# In[52]:


for i in range(len(types)) :
    if types[i] == 'cat' :
        vc = df_final[col[i]].value_counts()
        plt.bar(vc.index, vc.values)
        plt.xticks(rotation=90)
        #ax.set_xticklabels(df['Names'], rotation=90)
        plt.title('Tri à plat variable ' + col[i])
        plt.savefig('tri/' + col[i] + '.png')
        plt.show()
    if types[i] == 'int' :
        df_final.boxplot(col[i])
        plt.title('Boxplot variable ' + col[i])
        plt.savefig('tri/' + col[i] + '.png')
        plt.show()


# In[ ]:





# In[20]:


df_final.info()


# In[ ]:





# In[110]:


df_final.info()


# In[105]:


train_sample.info()


# In[106]:


m2.info()


# In[107]:


m2[m2.site_id.isnull()]


# In[98]:


building_sample


# In[94]:


building_sample


# In[93]:


m1


# In[92]:


weather_sample


# ### Pyspark

# In[ ]:




