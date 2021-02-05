#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from tensorflow.keras import metrics
from tensorflow import keras
import os
import tensorflow as tf
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle as pkl
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional, Dropout, concatenate, SpatialDropout1D, GlobalMaxPooling1D, Reshape, MaxPooling1D, Flatten, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras import Input

df = pd.read_csv("Train/cleaned_bureau.csv")
tdf = pd.read_csv("Test/cleaned_bureau.csv")
train_df = pd.read_csv("Train/cleaned_train.csv")
test_df = pd.read_csv("Test/cleaned_train.csv")
df.head()


# In[2]:


print(df.ID.value_counts(sort=True, ascending=False), df.shape)


# In[3]:


tdf.ID.value_counts(sort=True, ascending=False)


# In[4]:


cat_cols = ['SELF-INDICATOR', 'MATCH-TYPE', 'ACCT-TYPE', 'CONTRIBUTOR-TYPE',
       'OWNERSHIP-IND', 'ACCOUNT-STATUS', 'INSTALLMENT-TYPE',
       'ASSET_CLASS', 'INSTALLMENT-FREQUENCY',
       'DPD - HIST']    #should not be here
date_cols = ['DATE-REPORTED',  'DISBURSED-DT', 'CLOSE-DT', 
             'LAST-PAYMENT-DATE']
reg_cols = ['CREDIT-LIMIT/SANC AMT', 'DISBURSED-AMT/HIGH CREDIT', 'INSTALLMENT-AMT', 'CURRENT-BAL',
        'OVERDUE-AMT', 'WRITE-OFF-AMT', 'TENURE'] # , 'DPD - HIST']
array_cols = ['REPORTED DATE - HIST', 'CUR BAL - HIST',
       'AMT OVERDUE - HIST', 'AMT PAID - HIST']


# In[5]:


len(cat_cols) + len(date_cols) + len(reg_cols) + len(array_cols), df.shape


# In[6]:


assert all([df[i].dtype in ("bool" ,"object") for i in cat_cols])


# In[7]:


assert all([df[i].dtype == "float64" for i in reg_cols])


# In[8]:


df[date_cols].isnull().sum()


# In[9]:


for col in date_cols:
    df[col] = pd.to_datetime(df[col])
    tdf[col] = pd.to_datetime(tdf[col])
df[date_cols].isnull().sum()


# In[10]:


def get_length(amt):    
    if not pd.isnull(amt):
        return len(amt.split(","))
    else: 0
        
for col in array_cols:
    print("max train {}".format(df[col].apply(lambda x: get_length(x)).max()))
    
    print("max test {}".format(tdf[col].apply(lambda x: get_length(x)).max()))


# In[11]:


max_window = 420
ts_feature_vector = 205
max_array_size = 42
label_encoder_dict = {}
num_cores = 7

def encode_reg_cols(x):
    if pd.isnull(x):
        return 0.
    return np.float64(x)

def encode_array_cols(x):
    if pd.isnull(x):
        return [0] * max_array_size
    else:
        ret = []
        for val in x.split(","):
            try:
                ret.append(np.float32(val))
            except:
                ret.append(0.)
#         while len(ret) < max_array_size:
#             ret.insert(0, 0.)
        return ret


def encode_date_cols(x):
    if pd.isnull(x):
        return [-1., -1., -1., -1., -1.]
    else:
        return np.array([x.hour, x.minute, x.day, x.month, x.year], dtype=np.float64)


def encode_cat_cols(x, col):

    if pd.isnull(x): x = str(x)
    return np.array(label_encoder_dict[col].transform([x]), dtype=np.float64)


# In[12]:




for col in tqdm(cat_cols):
    if col not in label_encoder_dict:
        label_encoder_dict[col] = LabelEncoder()
    print(col)
    label_encoder_dict[col].fit(df[col].append(tdf[col]).fillna("nan"))
    


# In[13]:


temp_df = df[df.ID == 141732]


# In[14]:


def encode_df_for_user(dframe):
    
    final = []
    for index, row in dframe.iterrows():
        
        l = []
        ret = np.array([], dtype=np.float64)
    
        # 10 * 1
        for col in cat_cols:
#             print(ret, encode_cat_cols(row[col], col))
            ret = np.concatenate((ret, encode_cat_cols(row[col], col)))
        
        # 7 * 1 = 7
        for col in reg_cols:
#             print(np.array([encode_reg_cols(row[col])]))
#             print(ret)
            ret = np.concatenate((ret, np.array([encode_reg_cols(row[col])])))
        
        # 5 * 4 = 20
        for col in date_cols:
            ret = np.concatenate((ret, encode_date_cols(row[col])))
            
        
        # 4 * 42 = 168
        for col in array_cols:
            l.append(encode_array_cols(row[col]))
            
        ret = np.concatenate((ret, np.array(tf.keras.preprocessing.sequence.pad_sequences(l, maxlen=max_array_size, padding='pre')).flatten()))
        assert len(ret) == ts_feature_vector, print(len(l), len(l[0]))
        final.append(ret)
    while len(final) < max_window:
        final.insert(0, [0.] * ts_feature_vector)
#     assert len(final) == max_window
    return np.array(final)


# In[15]:


encode_df_for_user(temp_df).shape


# In[16]:


epochs = 100
batch_size = 32


# In[17]:


train_label_encoders = {}
target_encoder = LabelEncoder()



train_cat_cols = ['Frequency', 'InstlmentMode', 'LoanStatus', 'PaymentMode', 'BranchID', 'Area', 
            'ManufacturerID', 'SupplierID', 'SEX', 'City', 'State', 'ZiPCODE']
target_col = ['Top-up Month']
train_reg_cols = ['AmountFinance', 'DisbursalAmount', 'EMI', 'AssetID', 'MonthlyIncome', 'Tenure', 'AssetCost', 'LTV', 'AGE']
train_date_cols = ['DisbursalDate', 'MaturityDAte', 'AuthDate']

for col in train_date_cols:
    train_df[col] = pd.to_datetime(train_df[col], errors="coerce")
    test_df[col] = pd.to_datetime(test_df[col], errors="coerce")
    
for col in tqdm(train_cat_cols):
    if col not in train_label_encoders:
        train_label_encoders[col] = LabelEncoder()
    print(col)
    fill_val = -1 if train_df[col].dtype == "int64" else "nan"
    if col == target_col[0]:
        train_label_encoders[col].fit(train_df[col].fillna(fill_val))

    else: train_label_encoders[col].fit(train_df[col].append(test_df[col]).fillna(fill_val))

target_encoder.fit(train_df[target_col])

def train_encode_cat_cols(x, col, tpe):

    if pd.isnull(x): 
        if tpe == "object": x = str(x)
        elif x == "int64": x = 0
        else: assert False
        
    return train_label_encoders[col].transform([x])   

def encode_target(x):
    return target_encoder.transform(x)


# In[ ]:





# In[18]:


train_max_len = 36
def generate_training_data(row):
    row = row[0]
    ret = []
    columns = ['ID', 'Frequency', 'InstlmentMode', 'LoanStatus', 'PaymentMode',
       'BranchID', 'Area', 'Tenure', 'AssetCost', 'AmountFinance',
       'DisbursalAmount', 'EMI', 'DisbursalDate', 'MaturityDAte', 'AuthDate',
       'AssetID', 'ManufacturerID', 'SupplierID', 'LTV', 'SEX', 'AGE',
       'MonthlyIncome', 'City', 'State', 'ZiPCODE']
    column_tpes = ['int64', 'object', 'object', 'object', 'object', 
                   'int64', 'object', 'int64', 'int64','float64', 
                   'float64', 'float64',  '<M8[ns]', '<M8[ns]', '<M8[ns]',
                   'int64', 'int64', 'int64', 'float64', 'object', 
                   'float64', 'float64', 'object', 'object', 'int64', 'object']
    
    for index in range(len(columns)):
        if columns[index] in train_cat_cols:
            
            ret.extend(train_encode_cat_cols(row[index], columns[index], column_tpes[index]))

        elif columns[index] in train_reg_cols:
            ret.append(encode_reg_cols(row[index]))

        elif columns[index] in train_date_cols:
            ret.extend(encode_date_cols(row[index]))
        else: pass
    return np.array(ret)
    
def generate_datasets_to_train(train_dframe, bureau_df, val_size=.2):
    ids = train_dframe["ID"].unique()
    np.random.shuffle(ids)
    sp = int((1. - val_size) * ids.shape[0])
    tr_ids, val_ids = ids[: sp], ids[sp:]
#     X_br, X_val_br = [], []
#     X, X_val = [], []
    y, y_val = [], []
    
    
    X = Parallel(n_jobs=num_cores)(delayed(generate_training_data)(train_dframe[train_dframe.ID == i].to_numpy()) for i in tqdm(tr_ids, total=len(tr_ids)))
    X_br = Parallel(n_jobs=num_cores)(delayed(encode_df_for_user)(bureau_df[bureau_df.ID == i]) for i in tqdm(tr_ids, total=len(tr_ids)))
    
    X_val = Parallel(n_jobs=num_cores)(delayed(generate_training_data)(train_dframe[train_dframe.ID == i].to_numpy()) for i in tqdm(val_ids, total=len(val_ids)))
    X_val_br = Parallel(n_jobs=num_cores)(delayed(encode_df_for_user)(bureau_df[bureau_df.ID == i]) for i in tqdm(val_ids, total=len(val_ids)))
    

    for i in tqdm(tr_ids):
        
#         X = Parallel(n_jobs=num_cores)(delayed(generate_training_data)(train_dframe[train_dframe.ID == i].to_numpy()))
        
        
#         X_br.append(encode_df_for_user(bureau_df[bureau_df.ID == i]))
#         X.append(generate_training_data(train_dframe[train_dframe.ID == i].to_numpy()))
        y.append(target_encoder.transform(train_dframe[train_dframe.ID == i][target_col].values))
        
    for i in tqdm(val_ids):
#         X_val_br = Parallel(n_jobs=num_cores)(delayed(encode_df_for_user)(bureau_df[bureau_df.ID == i]))
#         X_val = Parallel(n_jobs=num_cores)(delayed(generate_training_data)(train_dframe[train_dframe.ID == i].to_numpy()))
        
        
#         X_val_br.append(encode_df_for_user(bureau_df[bureau_df.ID == i]))
        
#         X_val.append(generate_training_data(train_dframe[train_dframe.ID == i].to_numpy()))
        y_val.append(target_encoder.transform(train_dframe[train_dframe.ID == i][target_col].values))
    
    
    return np.array(X), np.array(X_val), np.array(X_br), np.array(X_val_br), np.array(y), np.array(y_val)
    

    
    
    


# In[19]:


z = generate_datasets_to_train(train_df.head(64000), df)


# In[20]:


z[0].shape, z[1].shape, z[2].shape, z[3].shape, z[4].shape, z[5].shape


# In[21]:


pkl.dump(z, open("data.pkl", "wb"))


# In[32]:


# def get_model():
#     train_in = Input(shape=(train_max_len, ))
#     train_int = Dense(32,activation="relu")(train_in)
#     train_int = Dropout(.2, seed=42)(train_int)
#     bureau_in = Input(shape=(max_window, ts_feature_vector))
#     bureau_int = LSTM(128, kernel_initializer='he_uniform', return_sequences=True)(bureau_in)
#     bureau_int = LSTM(64, kernel_initializer='he_uniform', return_sequences=True)(bureau_int)
#     bureau_int = LSTM(36, kernel_initializer='he_uniform', return_sequences=True)(bureau_int)
#     bureau_int = Reshape((420*36,), input_shape=(None, 420, 36))(bureau_int)
#     x = concatenate([train_int, bureau_int])
#     x = Dropout(.2, seed=42)(x)
    
#     x = Dense(32,activation="relu")(x)
#     output = Dense(1, activation="sigmoid")(x)
#     model = Model([train_in, bureau_in], output)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         loss='categorical_crossentropy',
#         metrics=["acc"],
#     )
#     print(model.summary())
#     return model

# model = get_model()

# print(np.unique(z[4]))
# model.fit(x=[z[0], z[2]], y=z[4], validation_data=([z[1], z[3]], z[5]), epochs=10, batch_size=batch_size, shuffle=True)


# In[ ]:




