#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_file = 'final_features.csv'
super_train = 'super_train.csv'


# In[ ]:


df = pd.read_csv(train_file, index_col='ID', parse_dates=['Date'])


# In[ ]:


tdf = pd.read_csv(super_train, index_col='ID')


# In[ ]:


# print([i for i in df.columns if 'Open' == i])
# print([i for i in df.columns if 'High' == i])
# print([i for i in df.columns if 'Low' == i])
# print([i for i in df.columns if 'holiday' == i])


# In[ ]:


df = df.join(tdf[['Open', 'High', 'Low', 'holiday']])


# In[ ]:


def expand_df(dframe):
    dFrame = dframe.copy()
    dFrame['day'] = dFrame.Date.apply(lambda x: x.day)
    dFrame['month'] = dFrame.Date.apply(lambda x: x.month)
    dFrame['year'] = dFrame.Date.apply(lambda x: x.year)
    dFrame['dayofweek'] = dFrame.Date.apply(lambda x: x.dayofweek)
    dFrame['dayofyear'] = dFrame.Date.apply(lambda x: x.dayofyear)
    dFrame['weekofyear'] = dFrame.Date.apply(lambda x: x.weekofyear)
    return dFrame


# In[ ]:


df = expand_df(df)


# In[ ]:


# for col in df.columns:
#     if df[col].isna().sum() != 0: print(col)


# In[ ]:


# df.columns.tolist()


# In[ ]:


cat_cols = [ 'holiday','stock',
 'day',
 'month',
 'year',
 'dayofweek',
 'dayofyear',
 'weekofyear']


# In[ ]:


encoder = LabelEncoder()
for col in tqdm(cat_cols):
    df[col] = encoder.fit_transform(df[col])


# In[ ]:


df['train'] = df['Close'].apply(lambda x: not pd.isna(x))


# In[ ]:


# df


# In[ ]:


# df[['Close', 'stock']][df['train'] == True]


# In[ ]:


X, y = df.drop(columns=['Close', 'train'], axis=1)[df['train'] == True], df[['Close', 'stock']][df['train'] == True]


# In[ ]:


# X


# In[ ]:


# y


# In[ ]:





# In[ ]:





# In[ ]:


model_store = [0] * 103
metrics = [0] * 103
grid = {'learning_rate': [.1, .2, .3, .4], 'depth': [1, 2, 3, 4, 5]}
for stock in tqdm(X.stock.unique(), total=103):
    X_tr, y_tr = X[X['stock'] == stock], y[y['stock'] == stock]['Close']
#     model_store[stock] = CatBoostRegressor(loss_function='RMSE', depth=3, learning_rate=0.4, iterations=1000, 
#         random_seed=18, 
#         od_type='Iter',
#         od_wait=20,
#     )
#     model_store[stock].fit(
#         X_tr, y_tr, use_best_model=True,
#         cat_features=cat_cols,
#         eval_set=(X_val, y_val),
#         verbose=False,  
#         plot=True,
#     )
    model_store[stock] = CatBoostRegressor(loss_function='RMSE', 
                                           iterations=1000, 
                                            random_seed=18, 
                                            od_type='Iter',
                                            od_wait=20,
                                            verbose=False,
                                           task_type="GPU", 
                                           devices='0:1',
                                           cat_features=cat_cols
                        )
    metrics[stock] = model_store[stock].grid_search(
        grid,
        X=X_tr, y=y_tr, # use_best_model=True,
        
        shuffle=False,
        stratified=False,
        verbose=False,  
        plot=False
    )


# In[ ]:


for i in range(len(model_store)): 
    print(model_store[i].best_score_, metrics[i])


# In[ ]:


X_test, y_test = df.drop(columns=['Close', 'train'], axis=1)[df['train'] == False], df[['Close', 'stock']][df['train'] == False]


# In[ ]:


preds = []
for stock in tqdm(X_test.stock.unique()):
    preds.append(pd.DataFrame({'ID': X_test[X_test['stock'] == stock].index, 'Close': model_store[stock].predict(X_test[X_test['stock'] == stock])}))


# In[ ]:


pd.concat(preds).to_csv('result.csv', index=False)


# In[ ]:


# from IPython.display import FileLink


# In[ ]:


# FileLink('result.csv')


# In[ ]:





# In[ ]:





# In[ ]:




