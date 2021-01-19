#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ### 1. Import du jeu de données

# In[2]:


# Names of the files we import
f_train = 'train_sample.csv'
f_building = 'building_metadata_sample.csv'
f_weather = 'weather_sample.csv'


# In[3]:


# Import of data
train_raw = pd.read_csv('data/' + f_train)
building_meta_raw = pd.read_csv('data/' + f_building)
weather_raw = pd.read_csv('data/' + f_weather)


# ### 2. Division du timestamp

# In[4]:


from datetime import datetime


# In[5]:


train = train_raw.copy()


# In[6]:


train['annee'] = train.apply(lambda x : int(x['timestamp'][:4]), axis=1)


# In[7]:


train['mois'] = train.apply(lambda x : int(x['timestamp'][5:7]), axis=1)


# In[8]:


train['jour'] = train.apply(lambda x : int(x['timestamp'][8:10]), axis=1)


# In[9]:


train['weekday'] = train.apply(lambda x : datetime(int(x['timestamp'][:4]),int(x['timestamp'][5:7]),int(x['timestamp'][8:10])).strftime("%A"), axis=1)


# In[10]:


train['hour'] = train.apply(lambda x : x['timestamp'][11:16], axis=1)


# In[11]:


train.head()


# ### 3. Gestion des valeurs manquantes

# In[12]:


# Suppression de year_built et floor_count
building_meta = building_meta_raw.drop(['year_built', 'floor_count'], axis=1)


# In[13]:


# Suppression de cloud_coverage, dew_temperature et precip_depth_1_hr
weather = weather_raw.drop(['cloud_coverage', 'dew_temperature', 'precip_depth_1_hr'], axis=1)
weather = weather.dropna()


# ### 4. Transformation de la variable à expliquer

# In[14]:


train['meter_log'] = np.log(train['meter_reading'] + 1)


# ### 5. Rassemblement des jeux de données

# In[15]:


# Tout rassembler en un jeu de données a bon nouveau de granularité pour la modélisation
temp = pd.merge(building_meta, weather, on='site_id', how='inner')
df = pd.merge(temp, train, on=['building_id', 'timestamp'], how='inner')


# In[16]:


df = df.drop(['timestamp'], axis=1)


# ### 5'. Conversion des types

# In[17]:


col_cat = ['primary_use', 'annee', 'mois', 'jour', 'weekday', 'hour']


# In[18]:


for col in col_cat :
    df[col] = df[col].astype('category')


# ### 6. Division du jdd par type de meter

# In[19]:


# Division en 4 jeux de données pour 4 modélisations en fonction du type de meter
df0 = df[df['meter'] == 0]
df1 = df[df['meter'] == 1]
df2 = df[df['meter'] == 2]
df3 = df[df['meter'] == 3]


# In[20]:


df_list = [df0, df1, df2, df3]


# ### 7. Gestion des outliers

# In[21]:


import matplotlib.pyplot as plt


# In[22]:


for i, d in enumerate(df_list) :
    plt.boxplot(d['meter_reading'])
    plt.title('Boxplot for energy ' + str(i))
    plt.show()


# In[23]:


df_list_no_outliers = []
for i, d in enumerate(df_list) :
    sd = d['meter_reading'].std()
    moy = d['meter_reading'].mean()
    print(i, 'before :', len(d))
    df_list_no_outliers.append(d[d['meter_reading'] < (moy + 4*sd)].drop('meter', axis=1))
    print(i, 'after :',len(df_list_no_outliers[i]))


# ### 8. Division en train/test

# In[24]:


y_list = []
y_list_log = []
X_list = []
for d in df_list_no_outliers :
    y_list.append(d['meter_reading'])
    y_list_log.append(d['meter_log'])
    X_list.append(d.drop(['meter_reading', 'meter_log'], axis=1))


# In[25]:


X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

X_train_log_list = []
X_test_log_list = []
y_train_log_list = []
y_test_log_list = []
for i in range(4) :
    X_train, X_test, y_train, y_test = train_test_split(X_list[i], y_list[i], test_size=0.33, random_state=42)
    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)
    
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_list[i], y_list_log[i], test_size=0.33, random_state=42)
    X_train_log_list.append(X_train_log)
    X_test_log_list.append(X_test_log)
    y_train_log_list.append(y_train_log)
    y_test_log_list.append(y_test_log)


# In[26]:


X_train_dummies = []
X_test_dummies = []
y_train_dummies = []
y_test_dummies = []
for i in range(4) :
    X_train, X_test, y_train, y_test = train_test_split(X_list[i], y_list[i], test_size=0.33, random_state=42)
    
    # We create the dummies for models that need it to work
    X_train = pd.get_dummies(X_train, prefix=col_cat)
    X_test = pd.get_dummies(X_test, prefix=col_cat)
    
    X_train_dummies.append(X_train)
    X_test_dummies.append(X_test)
    y_train_dummies.append(y_train)
    y_test_dummies.append(y_test)


# ### 10. Modélisation

# ##### 10.1 Catboost

# In[27]:


from catboost import CatBoostRegressor

iterations = [10,30,50]
learning_rates = [0.1,1,10]
depth = [2,4,6]

catboost_models = {}
catboost_preds = {}

num=0
for it in iterations :
    for lr in learning_rates :
        for de in depth :
            num+=1
            name = 'it_' + str(it) + '_lr_' + str(lr) + '_de_' + str(de)
            print('itération', num, '\n')
            catboost_models[name] = []
            catboost_preds[name] = []
            
            for i in range(4) :
                model = CatBoostRegressor(iterations=it, learning_rate=lr, depth=de, cat_features=col_cat)
                model.fit(X_train_list[i], y_train_list[i])
                preds = model.predict(X_test_list[i])
                print('')
                catboost_models[name].append(model)
                catboost_preds[name].append(preds)


# ##### 10.2 Catboost log

# In[ ]:


from catboost import CatBoostRegressor

iterations = [10,30,50]
learning_rates = [0.1,1]
depth = [2,4,6]

catboost_log_models = {}
catboost_log_preds = {}

num=0
for it in iterations :
    for lr in learning_rates :
        for de in depth :
            num+=1
            name = 'it_' + str(it) + '_lr_' + str(lr) + '_de_' + str(de)
            catboost_log_models[name] = []
            catboost_log_preds[name] = []
            print('itération', num, '\n')
            
            for i in range(4) :
                model = CatBoostRegressor(iterations=it, learning_rate=lr, depth=de, cat_features=col_cat)
                model.fit(X_train_log_list[i], y_train_log_list[i])
                preds = model.predict(X_test_log_list[i])
                print('')
                catboost_log_models[name].append(model)
                catboost_log_preds[name].append(preds)


# ##### 10.3 Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

max_depth = [2,5,8]
n_estimators = [10, 100, 1000]
criterion = ['mse', 'mae']

rfc_models = {}
rfc_preds = {}

num=0
for md in max_depth :
    for ne in n_estimators :
        for cr in criterion :
            num+=1
            name = 'md_' + str(md) + '_ne_' + str(ne) + '_cr_' + str(cr)
            rfc_models[name] = []
            rfc_preds[name] = []
            
            
            for i in range(4) :
                model = RandomForestRegressor(max_depth=md, n_estimators=ne, criterion=cr, random_state=0)
                model.fit(X_train_dummies[i], y_train_dummies[i])
                preds = model.predict(X_test_dummies[i])
                rfc_models[name].append(model)
                rfc_preds[name].append(preds)


# ##### 10.4 Random Forest log

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

max_depth = [2,5,8]
n_estimators = [10, 100, 1000]
criterion = ['mse', 'mae']

rfc_models_log = {}
rfc_preds_log = {}

num=0
for md in max_depth :
    for ne in n_estimators :
        for cr in criterion :
            num+=1
            name = 'md_' + str(md) + '_ne_' + str(ne) + '_cr_' + str(cr)
            rfc_models_log[name] = []
            rfc_preds_log[name] = []
            
            for i in range(4) :
                model = RandomForestRegressor(max_depth=md, n_estimators=ne, criterion=cr, random_state=0)
                model.fit(X_train_dummies[i], y_train_log_list[i])
                preds = model.predict(X_test_dummies[i])
                rfc_models_log[name].append(model)
                rfc_preds_log[name].append(preds)


# ### 11. Evaluation des résultats

# In[29]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import json


# ##### 11.1 Catboost

# In[31]:


metrics_catboost = {}
for key in catboost_preds :
    metrics_catboost[key] = {'rmse_list' : [], 'r2_list' : []}
    for i in range(4) :
        rmse = mean_squared_error(y_test_list[i], catboost_preds[key][i])
        r2 = r2_score(y_test_list[i], catboost_preds[key][i])
        metrics_catboost[key]['rmse_list'].append(rmse)
        metrics_catboost[key]['r2_list'].append(r2)
        print(key, round(r2,2))


# ##### 11.2 Catboost log

# In[31]:


metrics_catboost_log = {}
for key in catboost_preds_log :
    metrics_catboost_log[key] = {'rmse_list' : [], 'r2_list' : []}
    for i in range(4) :
        rmse = mean_squared_error(y_test_list_log[i], catboost_preds_log[key][i])
        r2 = r2_score(y_test_log_list[i], catboost_preds_log[key][i])
        metrics_catboost_log[key]['rmse_list'].append(rmse)
        metrics_catboost_log[key]['r2_list'].append(r2)
        print(key, round(r2,2))


# ##### 11.3 Random Forest

# In[ ]:


metrics_rf = {}
for key in rfc_preds :
    metrics_rf[key] = {'rmse_list' : [], 'r2_list' : []}
    for i in range(4) :
        rmse = mean_squared_error(y_test_list[i], rfc_preds[key][i])
        r2 = r2_score(y_test_list[i], rfc_preds[key][i])
        metrics_rf[key]['rmse_list'].append(rmse)
        metrics_rf[key]['r2_list'].append(r2)
        print(key, round(r2,2))


# ##### 11.4 Random Forest log

# In[ ]:


metrics_rf_log = {}
for key in rfc_preds :
    metrics_rf_log[key] = {'rmse_list' : [], 'r2_list' : []}
    for i in range(4) :
        rmse = mean_squared_error(y_test_log[i], rfc_preds[key][i])
        r2 = r2_score(y_test_log[i], rfc_preds[key][i])
        metrics_rf_log[key]['rmse_list'].append(rmse)
        metrics_rf_log[key]['r2_list'].append(r2)
        print(key, round(r2,2))

