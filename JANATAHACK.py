#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import init_ops, math_ops
from tensorflow.python.keras.utils.generic_utils import to_list
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score  
import seaborn as sns
import codecs
import warnings 
warnings.filterwarnings("ignore") 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import coo_matrix, hstack, vstack

import re
from nltk.corpus import stopwords


# In[ ]:


# from google.colab import files
# uploaded = files.upload()
# # !mv 'train (1).csv'  train.csv


# In[ ]:





# In[ ]:


df = pd.read_csv("train.csv")


# In[ ]:


df.set_index("ID", inplace=True)


# In[ ]:


def data_text_preprocess(total_text, ind, col):
    # Remove int values from text data as that might not be imp
    if type(total_text) is not int:
        string = ""
        # replacing all special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # replacing multiple spaces with single space
        total_text = re.sub('\s+',' ', str(total_text))
        # bring whole text to same lower-case scale.
#         total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from text
            #if not word in stop_words:
            string += word + " "
        
        df[col][ind] = string


# In[ ]:


for index, row in df.iterrows():
    if type(row['ABSTRACT']) is str:
        data_text_preprocess(row['ABSTRACT'], index, 'ABSTRACT')
    if type(row['TITLE']) is str:
        data_text_preprocess(row['TITLE'], index, 'TITLE')
df.head(10)


# In[ ]:



train, test = train_test_split(df, random_state=42, test_size=0.00001, shuffle=True)
# train.shape, test.shape
# test = pd.read_csv("cleaned_test_data.csv")


# In[ ]:



train_words_title, test_words_title = set(), set()
train_words_abstract, test_words_abstract = set(), set()
for index, row in train.iterrows():
    for  i in row['ABSTRACT'].split():
        train_words_abstract.add(i)
    for i in row['TITLE'].split():
        train_words_title.add(i)

for index, row in test.iterrows():
    for  i in row['ABSTRACT'].split():
        test_words_abstract.add(i)
    for i in row['TITLE'].split():
        test_words_title.add(i)
len(train_words_abstract - test_words_abstract), len(train_words_title - test_words_title)


# In[ ]:


len(train_words_abstract), len(test_words_abstract), len(train_words_title), len(test_words_title)


# In[ ]:


# w2i_title, w2i_abstract = {}, {}
# cnt = 0
# for i in train_words_abstract:
#     w2i_abstract.add()


# In[ ]:





# In[ ]:


# classifiers = {
#     "LogisiticRegression": LogisticRegression(),
#     "KNearest": KNeighborsClassifier(),
#     "Support Vector Classifier": SVC(),
#     "DecisionTreeClassifier": DecisionTreeClassifier(),
#     "MultinimialNB": MultinomialNB()
# }


# In[ ]:


abstract_max_len = 3000
title_max_len = 300
def get_label(row):
    label = [0, 0, 0, 0, 0, 0]
    columns = ['Computer Science', 'Physics', 'Mathematics',
       'Statistics', 'Quantitative Biology', 'Quantitative Finance']
    for col in range(len(columns)):
        if row[columns[col]] == 1:
            label[col] = 1
    return np.array(label)

def get_data(dframe):
#     X_title_vectors = count_vectorizer.transform(dframe['TITLE'])
#     X_abstract_vectors = count_vectorizer2.transform(dframe['ABSTRACT'])
#     X_title_vectors = hstack([X_title_vectors, X_abstract_vectors]).toarray()
    X_dframe = []
    y_dframe = []
    for index, row in tqdm(dframe.iterrows()):
        new_abstract_seq = []
        new_title_seq = []
        abstract = row["ABSTRACT"].split()
        title = row["TITLE"].split()
        for i in range(abstract_max_len):
            try: new_abstract_seq.append(abstract[i])
            except: new_abstract_seq.append("__PAD__")
        
        for i in range(title_max_len):
            try: new_title_seq.append(title[i])
            except: new_title_seq.append("__PAD__")
        y_dframe.append(get_label(row))        
        X_dframe.append([new_abstract_seq, new_title_seq])
    y_dframe = np.array(y_dframe)
    print((len(X_dframe), len(X_dframe[0])), y_dframe.shape)
    return X_dframe, y_dframe


# In[ ]:


X, y = get_data(train)
X_test, y_test = get_data(test)


# In[ ]:


batch_size = 32


# In[ ]:


sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)


# In[ ]:


def ElmoEmbedding1(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(batch_size*[abstract_max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]
def ElmoEmbedding2(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(batch_size*[title_max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]


# In[ ]:


tf.compat.v1.disable_eager_execution()
elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.tables_initializer())


# In[ ]:


input_text1 = Input(shape=(title_max_len,), dtype="string", name="input")
embedding1 = Lambda(ElmoEmbedding2, output_shape=(100), name="embedding")(input_text1)
# input_text2 = Input(shape=(title_max_len,), dtype=tf.string)
# embedding2 = Lambda(ElmoEmbedding2, output_shape=(None, 1024))(input_text2)
# embedding = concatenate([embedding1, embedding2])
x = LSTM(64)(embedding1)
# x = LSTM(32)(x)

# # x_rnn = Bidirectional(LSTM(units=64, return_sequences=True
# #                            ), name="bi2")(x)

x_rnn = Dense(32, activation="relu", name="dense1")(x)
# x = add([x, x_rnn])  # residual connection to the first biLSTM
out = Dense(6, activation="sigmoid", name="dense")(x_rnn)

model = Model(inputs=input_text1, outputs=out)


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


len(X), len(X[0][0]), len(X[0][1]), len(y), len(y[0])


# In[ ]:


X1 = [X[i][1] for i in range(len(X))]
# total 655
X1 = X1[:655 * batch_size]
y1 = y[:655 * batch_size]
# X1_test = [X_test[i][1] for i in range(len(X_test))]
# X1_test = X1_test[:131 * batch_size]
# y1_test = y_test[:131 * batch_size]
history = model.fit(np.array(X1), y1, class_weight='balanced', batch_size=batch_size, epochs=3, verbose=1)


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score
# total 655
X1_test = X1[-25 * batch_size:]
y1_test = y[-25 * batch_size:]
def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
x_test = list(divide_chunks(X1_test, batch_size))

# y_test = divide_chunks(y_test, batch_size)
predictions = []
for x in tqdm(x_test):
  for i in model.predict(np.array(x)):
    predictions.append(i)


predictions = np.array(predictions)
thresholds=[.05, .04, .03, .02, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for val in thresholds:
    pred=predictions.copy()

    pred[pred>=val]=1
    pred[pred<val]=0
  
    precision = precision_score(y1_test, pred, average='micro')
    recall = recall_score(y1_test, pred, average='micro')
    f1 = f1_score(y1_test, pred, average='micro')
   
    print("Micro-average quality numbers for val " + str(val))
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))


# In[ ]:





# In[ ]:


tdf = pd.read_csv('test.csv')


# In[ ]:


tdf.shape


# In[ ]:


def data_test_preprocess(total_text):
    # Remove int values from text data as that might not be imp
    if type(total_text) is not int:
        string = ""
        # replacing all special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # replacing multiple spaces with single space
        total_text = re.sub('\s+',' ', str(total_text))
        # bring whole text to same lower-case scale.
#         total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from text
            #if not word in stop_words:
            string += word + " "
        
    return string
def get_test_data(dframe):
#     X_title_vectors = count_vectorizer.transform(dframe['TITLE'])
#     X_abstract_vectors = count_vectorizer2.transform(dframe['ABSTRACT'])
#     X_title_vectors = hstack([X_title_vectors, X_abstract_vectors]).toarray()
    X_dframe = []
    y_dframe = []
    for index, row in tqdm(dframe.iterrows()):
        new_abstract_seq = []
        new_title_seq = []
        abstract = row["ABSTRACT"].split()
        title = data_test_preprocess(row["TITLE"]).split()
        for i in range(abstract_max_len):
            try: new_abstract_seq.append(abstract[i])
            except: new_abstract_seq.append("__PAD__")
        
        for i in range(title_max_len):
            try: new_title_seq.append(title[i])
            except: new_title_seq.append("__PAD__")
        # y_dframe.append(get_label(row))        
        X_dframe.append(new_title_seq)
    # y_dframe = np.array(y_dframe)
    # print((len(X_dframe), len(X_dframe[0])), y_dframe.shape)
    return X_dframe

columns = ['Computer Science', 'Physics', 'Mathematics',
       'Statistics', 'Quantitative Biology', 'Quantitative Finance']
# for i in columns:
#   tdf[i] = pd.Series(np.zeros(len(tdf)))
X_test = get_test_data(tdf)


# In[ ]:


def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
x_test = list(divide_chunks(X_test, batch_size))
for i in range(3): x_test[-1].append(['__PAD__'] * title_max_len)
# y_test = divide_chunks(y_test, batch_size)
predictions = []
for x in tqdm(x_test):
  for i in model.predict(np.array(x)):
    predictions.append([1 if a >= .35 else 0 for a in i])
len(predictions)


# In[ ]:


for col in range(len(columns)):
    tdf[columns[col]] = pd.Series(np.array([i[col] for i in predictions[:-3]], dtype='int'))


# In[ ]:


np.array([i[0] for i in predictions], dtype='int').shape
tdf.tail()


# In[ ]:


predictions[2]


# In[ ]:


model.predict(np.array(x_test[0]))[2]


# In[ ]:


model.predict(np.array(x_test[0]))


# In[ ]:


tdf.drop(['TITLE', 'ABSTRACT'], axis=1).to_csv('test_submit.csv', index=False)


# In[ ]:


# from google.colab import files
# files.download('test_submit.csv') 


# In[ ]:





# In[ ]:




