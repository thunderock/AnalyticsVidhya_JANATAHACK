import numpy as np # linear algebra
import pandas as pd 
import os
from tqdm import tqdm
import tensorflow as tf
import swifter

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Train/cleaned_bureau.csv") 

tdf = pd.read_csv("Test/cleaned_bureau.csv")

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


for col in date_cols:
    df[col] = pd.to_datetime(df[col])
    tdf[col] = pd.to_datetime(tdf[col])


def get_length(amt):
    if not pd.isnull(amt):
        return len(amt.split(","))
    else: 0

for col in array_cols:
    print("max train {}".format(df[col].apply(lambda x: get_length(x)).max()))

    print("max test {}".format(tdf[col].apply(lambda x: get_length(x)).max()))

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
        return [x.hour, x.minute, x.day, x.month, x.year]


def encode_cat_cols(x, col):
    return label_encoder_dict[col].transform(x.fillna("nan").values)




for col in tqdm(cat_cols):
    if col not in label_encoder_dict:
        label_encoder_dict[col] = LabelEncoder()
    print(col)
    label_encoder_dict[col].fit(df[col].append(tdf[col]).fillna("nan"))

final_columns = ["ID"]
for col in array_cols:
    temp = []
    for i in range(42):
        temp.append(col + '_' + str(i))
    final_columns += temp

for col in date_cols:
    final_columns += [col + "_0", col + "_1", col + "_2", col + "_3", col + "_4"]

for col in cat_cols:
    final_columns += [col]

for col in reg_cols:
    final_columns += [col]


len(final_columns), len(set(final_columns)), final_columns

final_df = pd.DataFrame(columns=final_columns)
for col in tqdm(df.columns):
    if col in reg_cols:
        final_df[col] = df[col].swifter.apply(encode_reg_cols)

    elif col in cat_cols:
        final_df[col] = encode_cat_cols(df[col], col=col)

    elif col in date_cols:
        final_df[col + '_0'], final_df[col + '_1'], final_df[col + '_2'], final_df[col + '_3'], final_df[col + '_4'] = \
        zip(*df[col].map(encode_date_cols))

    elif col in array_cols:
        final_df[col + '_0'], final_df[col + '_1'], final_df[col + '_2'], final_df[col + '_3'], final_df[col + '_4'], \
        final_df[col + '_5'], final_df[col + '_6'], final_df[col + '_7'], final_df[col + '_8'], final_df[col + '_9'],\
        final_df[col + '_10'], final_df[col + '_11'], final_df[col + '_12'], final_df[col + '_13'], final_df[col + '_14'],\
        final_df[col + '_15'], final_df[col + '_16'], final_df[col + '_17'], final_df[col + '_18'], final_df[col + '_19'],\
        final_df[col + '_20'], final_df[col + '_21'], final_df[col + '_22'], final_df[col + '_23'], final_df[col + '_24'],\
        final_df[col + '_25'], final_df[col + '_26'], final_df[col + '_27'], final_df[col + '_28'], final_df[col + '_29'],\
        final_df[col + '_30'], final_df[col + '_31'], final_df[col + '_32'], final_df[col + '_33'], final_df[col + '_34'],\
        final_df[col + '_35'], final_df[col + '_36'], final_df[col + '_37'], final_df[col + '_38'], final_df[col + '_39'],\
        final_df[col + '_40'], final_df[col + '_41'] = zip(*df[col].map(encode_array_cols))

    else:
        final_df[col] = df[col]




final_df.to_csv("final_training.csv", index=False)

final_df = pd.DataFrame(columns=final_columns)
for col in tqdm(tdf.columns):
    if col in reg_cols:
        final_df[col] = tdf[col].swifter.apply(encode_reg_cols)

    elif col in cat_cols:
        final_df[col] = encode_cat_cols(tdf[col], col=col)

    elif col in date_cols:
        final_df[col + '_0'], final_df[col + '_1'], final_df[col + '_2'], final_df[col + '_3'], final_df[col + '_4'] = \
        zip(*tdf[col].map(encode_date_cols))

    elif col in array_cols:
        final_df[col + '_0'], final_df[col + '_1'], final_df[col + '_2'], final_df[col + '_3'], final_df[col + '_4'], \
        final_df[col + '_5'], final_df[col + '_6'], final_df[col + '_7'], final_df[col + '_8'], final_df[col + '_9'],\
        final_df[col + '_10'], final_df[col + '_11'], final_df[col + '_12'], final_df[col + '_13'], final_df[col + '_14'],\
        final_df[col + '_15'], final_df[col + '_16'], final_df[col + '_17'], final_df[col + '_18'], final_df[col + '_19'],\
        final_df[col + '_20'], final_df[col + '_21'], final_df[col + '_22'], final_df[col + '_23'], final_df[col + '_24'],\
        final_df[col + '_25'], final_df[col + '_26'], final_df[col + '_27'], final_df[col + '_28'], final_df[col + '_29'],\
        final_df[col + '_30'], final_df[col + '_31'], final_df[col + '_32'], final_df[col + '_33'], final_df[col + '_34'],\
        final_df[col + '_35'], final_df[col + '_36'], final_df[col + '_37'], final_df[col + '_38'], final_df[col + '_39'],\
        final_df[col + '_40'], final_df[col + '_41'] = zip(*tdf[col].map(encode_array_cols))

    else:
        final_df[col] = tdf[col]




final_df.to_csv("final_testing.csv", index=False)


