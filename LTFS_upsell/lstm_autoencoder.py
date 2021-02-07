#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from tensorflow.keras import metrics
from tensorflow import keras
import os, logging
import tensorflow as tf
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle as pkl
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional, Dropout, concatenate, SpatialDropout1D, GlobalMaxPooling1D, Reshape, MaxPooling1D, Flatten, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow.keras.backend as K

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
df = pd.read_csv("Train/cleaned_bureau.csv")
tdf = pd.read_csv("Test/cleaned_bureau.csv")
train_df = pd.read_csv("Train/cleaned_train.csv")
test_df = pd.read_csv("Test/cleaned_train.csv")

df.head()


# In[ ]:


df.ID.value_counts(sort=True, ascending=False), tdf.shape


# In[ ]:


tdf.ID.value_counts(sort=True, ascending=False)


# In[ ]:


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


# In[ ]:


len(cat_cols) + len(date_cols) + len(reg_cols) + len(array_cols), df.shape


# In[ ]:


assert all([df[i].dtype in ("bool" ,"object") for i in cat_cols])


# In[ ]:


assert all([df[i].dtype == "float64" for i in reg_cols])


# In[ ]:


df[date_cols].isnull().sum()


# In[ ]:


for col in date_cols:
    df[col] = pd.to_datetime(df[col])
    tdf[col] = pd.to_datetime(tdf[col])
df[date_cols].isnull().sum()


# In[ ]:


def get_length(amt):    
    if not pd.isnull(amt):
        return len(amt.split(","))
    else: 0
        
for col in array_cols:
    print("max train {}".format(df[col].apply(lambda x: get_length(x)).max()))
    
    print("max test {}".format(tdf[col].apply(lambda x: get_length(x)).max()))


# In[ ]:


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
    ret = [0] * max_array_size
    if not pd.isnull(x):
        l = x.split(",")
        y = max_array_size - len(l)
        for index in range(len(l)):
            try:
                ret[y + index] = np.float32(l[index])
            except:
                ret[y + index] = 0.
    return ret


def encode_date_cols(x):
    if pd.isnull(x):
        return [-1., -1., -1., -1., -1.]
    else:
        return np.array([x.hour, x.minute, x.day, x.month, x.year], dtype=np.float64)


def encode_cat_cols(x, col):

    if pd.isnull(x): x = str(x)
    return np.array(label_encoder_dict[col].transform([x]), dtype=np.float64)


# In[ ]:




for col in tqdm(cat_cols):
    if col not in label_encoder_dict:
        label_encoder_dict[col] = LabelEncoder()
    print(col)
    label_encoder_dict[col].fit(df[col].append(tdf[col]).fillna("nan"))
    


# In[ ]:


# temp_df = df[df.ID == 141732]


# In[ ]:


def encode_df_for_user(dframe):
    
    final = []
    for index, row in dframe.iterrows():
        
        l = [None] * 4
        ret = np.array([], dtype=np.float64)
    
        # 10 * 1
        for col in cat_cols:
            ret = np.concatenate((ret, encode_cat_cols(row[col], col)))
        
        # 7 * 1 = 7
        for col in reg_cols:
            ret = np.concatenate((ret, np.array([encode_reg_cols(row[col])])))
        
        # 5 * 4 = 20
        for col in date_cols:
            ret = np.concatenate((ret, encode_date_cols(row[col])))
        
        # 4 * 42 = 168
        for i in range(4):
            l[i] = encode_array_cols(array_cols[i])
            
        
        ret = np.concatenate((ret, np.array(tf.keras.preprocessing.sequence.pad_sequences(l, maxlen=max_array_size, padding='pre')).flatten()))
        assert len(ret) == ts_feature_vector, print(len(l), len(l[0]))
        final.append(ret)
    while len(final) < max_window:
        final.insert(0, [0.] * ts_feature_vector)
    return np.array(final)


# In[ ]:


# encode_df_for_user(temp_df).shape


# In[ ]:


epochs = 100
batch_size = 128


# In[ ]:


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





# In[ ]:


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
    y, y_val = [], []
    
    
    X = Parallel(n_jobs=num_cores)(delayed(generate_training_data)(train_dframe[train_dframe.ID == i].to_numpy()) for i in tqdm(tr_ids, total=len(tr_ids)))
    X_br = Parallel(n_jobs=num_cores)(delayed(encode_df_for_user)(bureau_df[bureau_df.ID == i]) for i in tqdm(tr_ids, total=len(tr_ids)))
    
    X_val = Parallel(n_jobs=num_cores)(delayed(generate_training_data)(train_dframe[train_dframe.ID == i].to_numpy()) for i in tqdm(val_ids, total=len(val_ids)))
    X_val_br = Parallel(n_jobs=num_cores)(delayed(encode_df_for_user)(bureau_df[bureau_df.ID == i]) for i in tqdm(val_ids, total=len(val_ids)))
    

    for i in tqdm(tr_ids):
        y.append(target_encoder.transform(train_dframe[train_dframe.ID == i][target_col].values))
        
    for i in tqdm(val_ids):
        y_val.append(target_encoder.transform(train_dframe[train_dframe.ID == i][target_col].values))
    return np.array(X), np.array(X_val), np.array(X_br), np.array(X_val_br), np.array(y), np.array(y_val)
    

def generate_datasets_to_train_for_one_user(train_dframe, bureau_df):
    
    X_br = []
    X = []
    
    X_br.append(encode_df_for_user(bureau_df))
    X.append(generate_training_data(train_dframe.to_numpy()))
    return np.array(X), np.array(X_br)
    
def generate_datasets_to_train_for_one_user_test(train_dframe, bureau_df):
    
    X_br =  []
    X = []
    X_br.append(encode_df_for_user(bureau_df))
    X.append(generate_training_data(train_dframe.to_numpy()))
    return np.array(X), np.array(X_br)


# In[ ]:


class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, list_ids, bs=batch_size, test=False):
        self.bs = bs
        self.list_ids = list_ids
        self.n_classes = 7
        self.shuffle = False if test else True
        self.test = test
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.bs))
    
    def __getitem__(self, index):
        idx_min = index * self.bs
        idx_max = min(idx_min + self.bs, len(self.list_ids))
        indexes = self.indexes[idx_min: idx_max]
        
        temp_list_ids = [self.list_ids[k] for k in indexes]
        if self.test:
            X = self.__data_generator(temp_list_ids)
            return X
        else:
            X, y = self.__data_generator(temp_list_ids)
            return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __data_generator(self, temp_list):
        X = [None] * len(temp_list)
        y = [None] * len(temp_list)
        for index in range(len(temp_list)):
            i = temp_list[index]
            if self.test:
                X[index] = generate_datasets_to_train_for_one_user_test(
                    test_df[test_df.ID == i],
                    tdf[tdf.ID == i]
                )
            else:
                X[index] = generate_datasets_to_train_for_one_user(
                    train_df[train_df.ID == i],
                    df[df.ID == i]
                )
                
                y[index] = train_df[train_df.ID == i][target_col].values[0]
                
        if self.test:
            return X

        y = target_encoder.transform(y)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    


# In[ ]:


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


# In[ ]:


def get_model():
    train_in = Input(shape=(train_max_len, ))
    train_int = Dense(32,activation="relu")(train_in)
    train_int = Dropout(.2, seed=42)(train_int)
    bureau_in = Input(shape=(max_window, ts_feature_vector))
    bureau_int = LSTM(128, kernel_initializer='he_uniform', return_sequences=True)(bureau_in)
    bureau_int = LSTM(64, kernel_initializer='he_uniform', return_sequences=True)(bureau_int)
    bureau_int = LSTM(36, kernel_initializer='he_uniform', return_sequences=True)(bureau_int)
    bureau_int = Reshape((420*36,), input_shape=(None, 420, 36))(bureau_int)
    x = concatenate([train_int, bureau_int])
    x = Dropout(.2, seed=42)(x)
    
    x = Dense(32,activation="relu")(x)
    output = Dense(7, activation="softmax")(x)
    model = Model([train_in, bureau_in], output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=f1_loss,
        metrics=["acc", f1_loss],
    )
    print(model.summary())
    return model

model = get_model()


# In[ ]:


val_size = .2
ids = train_df["ID"].unique()
np.random.shuffle(ids)
sp = int((1. - val_size) * ids.shape[0])
tr_ids, val_ids = ids[: sp], ids[sp:]
test_ids = test_df["ID"].unique()


# In[ ]:


train_gen = DataGenerator(tr_ids, bs=batch_size)
val_gen = DataGenerator(val_ids)
test_gen = DataGenerator(test_ids, test=True)


# In[ ]:


checkpoint_path = 'model_save/model.keras'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=f1_loss, patience=3, verbose=0,
    mode="min", restore_best_weights=True
)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=f1_loss, factor=0.1, patience=3, verbose=0,
    mode='min', min_delta=0.0001, cooldown=0, min_lr=0
)


# In[ ]:


model.fit_generator(generator=train_gen, validation_data=val_gen, 
                    epochs=10,
                   use_multiprocessing=True, workers=7, callbacks=[cp_callback, early_stopping, plateau])


predictions = model.predict_generator(test_gen, verbose=1)
predictions.shape




len(train_gen)





