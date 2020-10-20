#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import pandas as pd
import tensorflow as tf
import transformers
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import pickle as pkl
import gc
import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# In[ ]:


max_length = 400
batch_size = 32
epochs = 3

df = pd.read_csv('Train.csv')

X = df.iloc[:, 1: 6]
y = df.iloc[:, 6: 6 + 25]

LABELS = y.columns


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=True, test_size=.2)

train_X.shape, train_y.shape, test_X.shape, test_y.shape


# In[ ]:


del train_X, train_y
X


# In[ ]:


y


# In[ ]:



X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
# train_y = train_y.reset_index(drop=True)
test_y = test_y.reset_index(drop=True)
# train_X = train_X.reset_index(drop=True)
test_X = test_X.reset_index(drop=True)


# In[ ]:


test_X.head(5)


# In[ ]:


test_y


# In[ ]:


# train_y = train_y.values
test_y = test_y.values
y = y.values
test_y


# In[ ]:


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentences,
        depts,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentences = sentences
        self.depts = depts
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentences))
        self.on_epoch_end()

    def __len__(self):
        return len(self.sentences) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentences = self.sentences[indexes]

        encoded = self.tokenizer.batch_encode_plus(
            sentences.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        
        dept_ids = np.array(self.depts[indexes], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")
        
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, dept_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, dept_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


# In[ ]:


X.iloc[:,1:6].columns, y.shape


# In[ ]:


def reset_keras():
    sess = KK.get_session()
    KK.clear_session()
    sess.close()
    sess = KK.get_session()

#     try:
#         del classifier # this is from global space - change this as you need
#     except:
#         pass

    print(gc.collect()) # if it does something you should see a number as output

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))
    


# In[ ]:


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )

    input_departments = tf.keras.Input(
        shape = (4,), dtype=tf.float32, name='input_depts'
    )

    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )

    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )

    # Loading pretrained BERT model.
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")

    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False

    sequence_output, pooled_output = bert_model(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )

    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(sequence_output)

    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool, input_departments])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(25, activation="softmax")(dropout)
    model = tf.keras.models.Model(
        inputs=[input_ids, input_departments, attention_masks, token_type_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["acc", f1_m],
    )
    
print(f"Strategy: {strategy}")
model.summary()


# In[ ]:



val_data = BertSemanticDataGenerator(
    test_X['ABSTRACT'].values.astype("str"),
    test_X.iloc[:,1:6].values.astype("int32"),
    test_y,
    batch_size=batch_size,
    shuffle=False,
)


data = BertSemanticDataGenerator(
    X['ABSTRACT'].values.astype("str"),
    X.iloc[:,1:6].values.astype("int32"),
    y,
    batch_size=batch_size,
    shuffle=True,
)


# In[ ]:


X.iloc[:,1:6]


# In[ ]:


history = model.fit(
    data,
    validation_data=val_data,
    shuffle=True,
    epochs=epochs,
#     use_multiprocessing=True,
#     workers=-1,
)


# In[ ]:


K.clear_session()
# Unfreeze the bert_model.
bert_model.trainable = True
# Recompile the model to make the change effective.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["acc", f1_m],
)
model.summary()


# In[ ]:


batch_size = 4
val_data = BertSemanticDataGenerator(
    test_X['ABSTRACT'].values.astype("str"),
    test_X.iloc[:,1:6].values.astype("int32"),
    test_y,
    batch_size=batch_size,
    shuffle=False,
)


data = BertSemanticDataGenerator(
    X['ABSTRACT'].values.astype("str"),
    X.iloc[:,1:6].values.astype("int32"),
    y,
    batch_size=batch_size,
    shuffle=True,
)


# In[ ]:


history = model.fit(
    data,
    validation_data=val_data,
    epochs=epochs,
    #use_multiprocessing=True,
    #workers=-1,
)


# In[ ]:


val_data = BertSemanticDataGenerator(
    test_X['ABSTRACT'].values.astype("str"),
    test_X.iloc[:,1:6].values.astype("int32"),
    test_y,
    batch_size=1,
    shuffle=False,
)


preds = model.predict_generator(val_data, verbose=1, use_multiprocessing=True)


# In[ ]:


pkl.dump(test_y, open("val_original.pkl", "rb"))
pkl.dump(preds, open("val_preds.pkl", "rb"))
model.evaluate(val_data, verbose=1)


# In[ ]:


# def check_similarity(sentence, depts):
# #     sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
#     test_data = BertSemanticDataGenerator(
#         np.array(sentence), depts, labels=None, batch_size=1, shuffle=False, include_targets=False,
#     )

#     proba = model.predict(test_data)[0]
#     idx = np.argmax(proba)
#     proba = f"{proba[idx]: .2f}%"
#     pred = labels[idx]
#     return pred, proba


# In[ ]:


# s = train_X.iloc[0]['ABSTRACT']
# d = train_X.iloc[:,1:6].iloc[0].values
# check_similarity(s, d)
# # sentence2 = "Two women are standing with their eyes closed."
# # check_similarity(sentence1, sentence2)


# In[ ]:


# sentence1 = "A smiling costumed woman is holding an umbrella"
# sentence2 = "A happy woman in a fairy costume holds an umbrella"
# check_similarity(sentence1, sentence2)


# In[ ]:


# sentence1 = "A soccer game with multiple males playing"
# sentence2 = "Some men are playing a sport"
# check_similarity(sentence1, sentence2)


# In[ ]:


# model.save("model.h5")
# print("Saved model to disk")


# In[ ]:


tdf = pd.read_csv("Test.csv")
# df = df.iloc[:, 1: 6]
tdf.iloc[:, 2: 6].head(5)


# In[ ]:


tdf.shape


# In[ ]:



val_data = BertSemanticDataGenerator(
    tdf['ABSTRACT'].values.astype("str"),
    tdf.iloc[:,2:6].values.astype("int32"),
    # None,
    batch_size=2,
    labels=None,
    shuffle=False, 
    include_targets=False
)



# In[ ]:


preds = model.predict_generator(val_data, verbose=1, use_multiprocessing=True)


# In[ ]:


preds.shape, tdf.shape


# In[ ]:


cnt = {}
for x in y:
    ones = 0
    for xx in x:
        if xx == 1: ones += 1
    if ones in cnt: cnt[ones] += 1
    else: cnt[ones] = 1
for x in cnt: 
    print(x, cnt[x] / sum(cnt.values()))


# In[ ]:



THRESHOLD = .17


# In[ ]:


cnt2 = {}
for x in preds:
    ones = 0
    for xx in x:
        if xx > THRESHOLD: ones += 1
    if ones in cnt2: cnt2[ones] += 1
    else: cnt2[ones] = 1
for x in cnt2: 
    print(x, cnt2[x] / sum(cnt2.values()))


# In[ ]:





# In[ ]:


tdf.head(5)


# In[ ]:


for col in range(len(LABELS)):
    tdf[LABELS[col]] = [1 if x[col] > THRESHOLD else 0 for x in preds]
    


# In[ ]:


tdf.drop(columns=['ABSTRACT', 'Computer Science', 'Mathematics', 'Physics', 'Statistics']).to_csv("final.csv", index=False)


# In[ ]:



pkl.dump(preds, open('pred_proba.pkl', 'wb'))


# In[ ]:




