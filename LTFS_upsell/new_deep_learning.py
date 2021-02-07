#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from tensorflow.keras import metrics
from tensorflow import keras
import os, logging
import tensorflow as tf
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from joblib import Parallel, delayed
import pickle as pkl
from tensorflow.keras.layers import Concatenate, LSTM, Dense, TimeDistributed, Embedding, Bidirectional, Dropout, concatenate, SpatialDropout1D, GlobalMaxPooling1D, Reshape, MaxPooling1D, Flatten, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow.keras.backend as K

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




df = pd.read_csv("final_training.csv")
tdf = pd.read_csv("final_testing.csv")
train_df = pd.read_csv("Train/cleaned_train.csv")
test_df = pd.read_csv("Test/cleaned_train.csv")


cat_cols = ['SELF-INDICATOR', 'MATCH-TYPE', 'ACCT-TYPE', 'CONTRIBUTOR-TYPE',
       'OWNERSHIP-IND', 'ACCOUNT-STATUS', 'INSTALLMENT-TYPE',
       'ASSET_CLASS', 'INSTALLMENT-FREQUENCY',
       'DPD - HIST', 'DATE-REPORTED_0', 'DATE-REPORTED_1', 'DATE-REPORTED_2',
 'DATE-REPORTED_3','DATE-REPORTED_4','DISBURSED-DT_0',
 'DISBURSED-DT_1',
 'DISBURSED-DT_2',
 'DISBURSED-DT_3',
 'DISBURSED-DT_4',
 'CLOSE-DT_0',
 'CLOSE-DT_1',
 'CLOSE-DT_2',
 'CLOSE-DT_3',
 'CLOSE-DT_4',
 'LAST-PAYMENT-DATE_0',
 'LAST-PAYMENT-DATE_1',
 'LAST-PAYMENT-DATE_2',
 'LAST-PAYMENT-DATE_3',
 'LAST-PAYMENT-DATE_4']
reg_cols = ['REPORTED DATE - HIST_0', 'REPORTED DATE - HIST_1', 'REPORTED DATE - HIST_2', 'REPORTED DATE - HIST_3', 
            'REPORTED DATE - HIST_4', 'REPORTED DATE - HIST_5', 'REPORTED DATE - HIST_6', 'REPORTED DATE - HIST_7', 
            'REPORTED DATE - HIST_8', 'REPORTED DATE - HIST_9', 'REPORTED DATE - HIST_10', 'REPORTED DATE - HIST_11', 
            'REPORTED DATE - HIST_12', 'REPORTED DATE - HIST_13', 'REPORTED DATE - HIST_14', 'REPORTED DATE - HIST_15', 
            'REPORTED DATE - HIST_16', 'REPORTED DATE - HIST_17', 'REPORTED DATE - HIST_18', 'REPORTED DATE - HIST_19', 
            'REPORTED DATE - HIST_20', 'REPORTED DATE - HIST_21', 'REPORTED DATE - HIST_22', 'REPORTED DATE - HIST_23', 
            'REPORTED DATE - HIST_24', 'REPORTED DATE - HIST_25', 'REPORTED DATE - HIST_26', 'REPORTED DATE - HIST_27', 
            'REPORTED DATE - HIST_28', 'REPORTED DATE - HIST_29', 'REPORTED DATE - HIST_30', 'REPORTED DATE - HIST_31', 
            'REPORTED DATE - HIST_32', 'REPORTED DATE - HIST_33', 'REPORTED DATE - HIST_34', 'REPORTED DATE - HIST_35', 
            'REPORTED DATE - HIST_36', 'REPORTED DATE - HIST_37', 'REPORTED DATE - HIST_38', 'REPORTED DATE - HIST_39', 
            'REPORTED DATE - HIST_40', 'REPORTED DATE - HIST_41', 'CUR BAL - HIST_0', 'CUR BAL - HIST_1', 'CUR BAL - HIST_2', 
            'CUR BAL - HIST_3', 'CUR BAL - HIST_4', 'CUR BAL - HIST_5', 'CUR BAL - HIST_6', 'CUR BAL - HIST_7', 'CUR BAL - HIST_8', 
            'CUR BAL - HIST_9', 'CUR BAL - HIST_10', 'CUR BAL - HIST_11', 'CUR BAL - HIST_12', 'CUR BAL - HIST_13', 'CUR BAL - HIST_14', 
            'CUR BAL - HIST_15', 'CUR BAL - HIST_16', 'CUR BAL - HIST_17', 'CUR BAL - HIST_18', 'CUR BAL - HIST_19', 'CUR BAL - HIST_20', 
            'CUR BAL - HIST_21', 'CUR BAL - HIST_22', 'CUR BAL - HIST_23', 'CUR BAL - HIST_24', 'CUR BAL - HIST_25', 'CUR BAL - HIST_26', 
            'CUR BAL - HIST_27', 'CUR BAL - HIST_28', 'CUR BAL - HIST_29', 'CUR BAL - HIST_30', 'CUR BAL - HIST_31', 'CUR BAL - HIST_32', 
            'CUR BAL - HIST_33', 'CUR BAL - HIST_34', 'CUR BAL - HIST_35', 'CUR BAL - HIST_36', 'CUR BAL - HIST_37', 'CUR BAL - HIST_38', 
            'CUR BAL - HIST_39', 'CUR BAL - HIST_40', 'CUR BAL - HIST_41', 'AMT OVERDUE - HIST_0', 'AMT OVERDUE - HIST_1', 'AMT OVERDUE - HIST_2', 
            'AMT OVERDUE - HIST_3', 'AMT OVERDUE - HIST_4', 'AMT OVERDUE - HIST_5', 'AMT OVERDUE - HIST_6', 'AMT OVERDUE - HIST_7', 'AMT OVERDUE - HIST_8', 
            'AMT OVERDUE - HIST_9', 'AMT OVERDUE - HIST_10', 'AMT OVERDUE - HIST_11', 'AMT OVERDUE - HIST_12', 'AMT OVERDUE - HIST_13', 'AMT OVERDUE - HIST_14', 
            'AMT OVERDUE - HIST_15', 'AMT OVERDUE - HIST_16', 'AMT OVERDUE - HIST_17', 'AMT OVERDUE - HIST_18', 'AMT OVERDUE - HIST_19', 'AMT OVERDUE - HIST_20', 
            'AMT OVERDUE - HIST_21', 'AMT OVERDUE - HIST_22', 'AMT OVERDUE - HIST_23', 'AMT OVERDUE - HIST_24', 'AMT OVERDUE - HIST_25', 'AMT OVERDUE - HIST_26', 
            'AMT OVERDUE - HIST_27', 'AMT OVERDUE - HIST_28', 'AMT OVERDUE - HIST_29', 'AMT OVERDUE - HIST_30', 'AMT OVERDUE - HIST_31', 'AMT OVERDUE - HIST_32', 
            'AMT OVERDUE - HIST_33', 'AMT OVERDUE - HIST_34', 'AMT OVERDUE - HIST_35', 'AMT OVERDUE - HIST_36', 'AMT OVERDUE - HIST_37', 'AMT OVERDUE - HIST_38', 
            'AMT OVERDUE - HIST_39', 'AMT OVERDUE - HIST_40', 'AMT OVERDUE - HIST_41', 'AMT PAID - HIST_0', 'AMT PAID - HIST_1', 'AMT PAID - HIST_2', 'AMT PAID - HIST_3', 
            'AMT PAID - HIST_4', 'AMT PAID - HIST_5', 'AMT PAID - HIST_6', 'AMT PAID - HIST_7', 'AMT PAID - HIST_8', 'AMT PAID - HIST_9', 'AMT PAID - HIST_10', 'AMT PAID - HIST_11', 
            'AMT PAID - HIST_12', 'AMT PAID - HIST_13', 'AMT PAID - HIST_14', 'AMT PAID - HIST_15', 'AMT PAID - HIST_16', 'AMT PAID - HIST_17', 'AMT PAID - HIST_18', 
            'AMT PAID - HIST_19', 'AMT PAID - HIST_20', 'AMT PAID - HIST_21', 'AMT PAID - HIST_22', 'AMT PAID - HIST_23', 'AMT PAID - HIST_24', 'AMT PAID - HIST_25', 
            'AMT PAID - HIST_26', 'AMT PAID - HIST_27', 'AMT PAID - HIST_28', 'AMT PAID - HIST_29', 'AMT PAID - HIST_30', 'AMT PAID - HIST_31', 'AMT PAID - HIST_32', 
            'AMT PAID - HIST_33', 'AMT PAID - HIST_34', 'AMT PAID - HIST_35', 'AMT PAID - HIST_36', 'AMT PAID - HIST_37', 'AMT PAID - HIST_38', 'AMT PAID - HIST_39', 
            'AMT PAID - HIST_40', 'AMT PAID - HIST_41','CREDIT-LIMIT/SANC AMT', 'DISBURSED-AMT/HIGH CREDIT', 'INSTALLMENT-AMT', 'CURRENT-BAL',
        'OVERDUE-AMT', 'WRITE-OFF-AMT', 'TENURE']


# In[5]:


# len(cat_cols), len(reg_cols)


# In[6]:


# df[reg_cols].dtypes.value_counts()


# In[7]:


# df[cat_cols].dtypes.value_counts()


# In[8]:


# tdf[reg_cols].dtypes.value_counts()


# In[9]:


# tdf[cat_cols].dtypes.value_counts()


# In[10]:


max_window = 420
ts_feature_vector = 205
max_array_size = 42
label_encoder_dict = {}
num_cores = 7


# In[11]:


df.loc[:, df.columns != 'ID'].sample(10).values


# In[12]:


def encode_df_for_user(dframe):
    
    ret = dframe.loc[:, df.columns != 'ID'].values
    ret = np.pad(ret,(((max_window - ret.shape[0]),0),(0, 0)), 'constant')
#     ret = np.array(tf.keras.preprocessing.sequence.pad_sequences(ret, maxlen=max_window, padding='pre'))
    assert ret.shape == (max_window,205), ret.shape
    return ret
    
    


# In[13]:



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
    if train_df[col].dtype == "float64":
        train_df[col] = train_df[col].fillna(-1.).astype(np.int64)
        
    if test_df[col].dtype == "float64":
        test_df[col] = test_df[col].fillna(-1.).astype(np.int64)
    fill_val = -1 if train_df[col].dtype == "int64" else "nan"
    if col == target_col[0]:
        train_label_encoders[col].fit(train_df[col].fillna(fill_val))

    else: train_label_encoders[col].fit(train_df[col].append(test_df[col]).fillna(fill_val))

target_encoder.fit(train_df[target_col])

def train_encode_cat_cols(x, col, tpe):

    if pd.isnull(x): 
        if tpe == "object": x = str(x)
        elif tpe == "int64": x = -1
        else: assert False, x
        
    return train_label_encoders[col].transform([x])   

def encode_reg_cols(x):
    if pd.isnull(x):
        return -1.
    return np.float64(x)


def encode_cat_cols(x, col):

    if pd.isnull(x): x = str(x)
    return np.array(label_encoder_dict[col].transform([x]), dtype=np.float64)


def encode_target(x):
    return target_encoder.transform(x)


def encode_date_cols(x):
    if pd.isnull(x):
        return [-1, -1, -1, -1, -1]
    else:
        return np.array([x.hour, x.minute, x.day, x.month, x.year], dtype=np.float64)


# In[14]:


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


def generate_datasets_to_train_for_one_user(train_dframe, bureau_df):
    X_br = []
    X = []
    
    X_br = encode_df_for_user(bureau_df)
    X = generate_training_data(train_dframe.to_numpy())
    assert X_br.shape == (max_window, ts_feature_vector), X_br.shape
    
    assert X.shape == (train_max_len, ), X.shape
    return np.reshape(X, (1, X.shape[0])), np.reshape(X_br, (X_br.shape[0], 1, X_br.shape[1]))

def generate_datasets_to_train(train_dframe, bureau_df, val_size=.2):
    ids = train_dframe["ID"].unique()
    np.random.shuffle(ids)
    sp = int((1. - val_size) * ids.shape[0])
    tr_ids, val_ids = ids[: sp], ids[sp:]
    y, y_val = [None] * sp, [None] * int(ids.shape[0] * val_size)
    
    
    X = Parallel(n_jobs=num_cores)(delayed(generate_training_data)(train_dframe[train_dframe.ID == i].to_numpy()) for i in tqdm(tr_ids, total=len(tr_ids)))
    X_br = Parallel(n_jobs=num_cores)(delayed(encode_df_for_user)(bureau_df[bureau_df.ID == i]) for i in tqdm(tr_ids, total=len(tr_ids)))
    
    X_val = Parallel(n_jobs=num_cores)(delayed(generate_training_data)(train_dframe[train_dframe.ID == i].to_numpy()) for i in tqdm(val_ids, total=len(val_ids)))
    X_val_br = Parallel(n_jobs=num_cores)(delayed(encode_df_for_user)(bureau_df[bureau_df.ID == i]) for i in tqdm(val_ids, total=len(val_ids)))
    
    
    
    for i in tqdm(range(len(tr_ids))):
        assert train_dframe[train_dframe.ID == tr_ids[i]][target_col].values.shape[0] == 1
        y[i] = train_dframe[train_dframe.ID == tr_ids[i]][target_col].values[0]
    
    y = keras.utils.to_categorical(target_encoder.transform(y))
    for i in tqdm(range(len(val_ids))):
        assert train_dframe[train_dframe.ID == val_ids[i]][target_col].values.shape[0] == 1
        y_val[i] = train_dframe[train_dframe.ID == val_ids[i]][target_col].values[0]
    
    y_val = keras.utils.to_categorical(target_encoder.transform(y_val))
    
    return np.array(X), np.array(X_val), np.array(X_br), np.array(X_val_br), np.array(y), np.array(y_val)


# In[15]:


# z = generate_datasets_to_train(train_df.sample(10000), df)


# In[ ]:





# In[16]:


class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, list_ids, bs=64, test=False):
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
                X[index] = generate_datasets_to_train_for_one_user(
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
#         print(X)
        y = target_encoder.transform(y)
        y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


# In[17]:


val_size = .2
ids = train_df["ID"].unique()
np.random.shuffle(ids)
sp = int((1. - val_size) * ids.shape[0])
tr_ids, val_ids = ids[: sp], ids[sp:]


# In[18]:


train_gen = DataGenerator(tr_ids)
val_gen = DataGenerator(val_ids)


# In[19]:


def generate_datasets_to_test(train_dframe, bureau_df):
    tr_ids = train_dframe["ID"].unique()
    
    X = Parallel(n_jobs=num_cores)(delayed(generate_training_data)(train_dframe[train_dframe.ID == i].to_numpy()) for i in tqdm(tr_ids, total=len(tr_ids)))
    X_br = Parallel(n_jobs=num_cores)(delayed(encode_df_for_user)(bureau_df[bureau_df.ID == i]) for i in tqdm(tr_ids, total=len(tr_ids)))
    
    return np.array(X), np.array(X_br)


# In[20]:


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


# In[21]:


def get_model():
    train_in = Input(shape=(train_max_len, ))
    train_int = Dense(36,activation="relu")(train_in)
    train_int = Dropout(.2, seed=42)(train_int)
    bureau_in = Input(shape=(max_window, ts_feature_vector))
    bureau_int = LSTM(128, kernel_initializer='he_uniform', return_sequences=True)(bureau_in)
    bureau_int = LSTM(64, kernel_initializer='he_uniform', return_sequences=True)(bureau_int)
    bureau_int = LSTM(36, kernel_initializer='he_uniform', return_sequences=True)(bureau_int)
#     bureau_int = Reshape((420, 36,))(bureau_int)
#     bureau_int = Dense(32,activation="relu")(bureau_int)
    bureau_int = Flatten()(bureau_int)
    x = Concatenate()([train_int, bureau_int])
#     x = Merge([train_int, bureau_int], mode='concat')
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


checkpoint_path = 'model_save/model.keras'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=f1_loss, patience=3, verbose=0,
    mode="min", restore_best_weights=True
)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=f1_loss, factor=0.01, patience=3, verbose=0,
    mode='min', min_delta=0.0001, cooldown=0, min_lr=0
)


# In[23]:


# model.fit(x=[z[0], z[2]], y=z[4], validation_data=([z[1], z[3]], z[5]), 
#           epochs=2, batch_size=32, shuffle=True)
model.fit_generator(generator=train_gen, validation_data=val_gen, 
                    epochs=50,
                   use_multiprocessing=True, workers=7, callbacks=[TqdmCallback(verbose=1), cp_callback, early_stopping, plateau])

test_ids = test_df["ID"]

test_gen = DataGenerator(test_ids, bs= 1, test=True)


# In[32]:


predictions = model.predict_generator(test_gen, verbose=1)





sub = pd.DataFrame({"ID": test_df["ID"], target_col[0]: target_encoder.inverse_transform(np.argmax(predictions, axis=1))})




count = {}
for i in target_encoder.inverse_transform(np.argmax(predictions, axis=1)):
    if i not in count:
        count[i] = 0
    count[i] += 1
print(count)


# In[44]:


sub.to_csv("submission.csv", index=False)


# In[ ]:




