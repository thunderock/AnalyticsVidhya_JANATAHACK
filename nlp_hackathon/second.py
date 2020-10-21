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
import time
import sys

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

max_length = 400
batch_size = 64
epochs = 4

df = pd.read_csv('Train.csv')

X = df.iloc[:, 1: 6]
y = df.iloc[:, 6: 6 + 25]

LABELS = y.columns


train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=True, test_size=.2)
train_X.shape, train_y.shape, test_X.shape, test_y.shape


del train_X, train_y



X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
test_y = test_y.reset_index(drop=True)
test_X = test_X.reset_index(drop=True)


test_y = test_y.values
y = y.values



class BertSemanticDataGenerator(tf.keras.utils.Sequence):

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




# strategy = tf.distribute.MirroredStrategy()
# 
# with strategy.scope():

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
    loss="binary_crossentropy",
    metrics=["acc", f1_m],
)

model.summary()


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




print("training model")
history = model.fit(
    data,
    validation_data=val_data,
    shuffle=True,
    epochs=epochs,
)

K.clear_session()
gc.collect()
print("sleeping")
time.sleep(60)


bert_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["acc", f1_m],
)
model.summary()



epochs = 1
batch_size = 2
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

print("training trainable bert model")

history = model.fit(
    data,
    validation_data=val_data,
    epochs=epochs,
)


val_data = BertSemanticDataGenerator(
    test_X['ABSTRACT'].values.astype("str"),
    test_X.iloc[:,1:6].values.astype("int32"),
    test_y,
    batch_size=1,
    shuffle=False,
)


print("predicting validation data")
preds = model.predict_generator(val_data, verbose=1, use_multiprocessing=True)




print("evaluating validation data")
pkl.dump(test_y, open("val_original.pkl", "wb"))
pkl.dump(preds, open("val_preds.pkl", "wb"))
model.evaluate(val_data, verbose=1)


tdf = pd.read_csv("Test.csv")


val_data = BertSemanticDataGenerator(
    tdf['ABSTRACT'].values.astype("str"),
    tdf.iloc[:,2:6].values.astype("int32"),
    batch_size=2,
    labels=None,
    shuffle=False,
    include_targets=False
)


print("predicting test data")
preds = model.predict_generator(val_data, verbose=1, use_multiprocessing=True)


print("final predictions")
print(preds)

pkl.dump(preds, open('pred_proba.pkl', 'wb'))



pred = '/tmp/val_preds1.pkl'
y = '/tmp/val_original1.pkl'
final_pred = '/tmp/pred_proba1.pkl'


pred = pkl.load(open(pred, 'rb'))

y = pkl.load(open(y, 'rb'))
y = y[0:0 + pred.shape[0]]

thresholds = [0] * 25
counts = [0] * 25
y_trues = [0] * 25

def get_current_loss(p_col, true_col, th):
    p_col = sum(p_col > th)
    true_col = sum(true_col > th)
    return abs(p_col - true_col)

for col in range(25):
    current_loss = sys.maxsize
    temp_loss = sys.maxsize
    for threshold in np.linspace(0, 1, 20, endpoint=False):
        pred_col = np.take(pred, col, axis=1)
        y_col = np.take(y, col, axis=1)
        temp_loss = get_current_loss(pred_col, y_col, threshold)
        if  temp_loss < current_loss:
            thresholds[col] = threshold
            current_loss = temp_loss
            counts[col] = sum(pred_col > threshold)
            y_trues[col] = sum(y_col > threshold)
            # print(col, current_loss, thresholds[col], counts[col], y_trues[col])

print(thresholds)
print(counts)
print(y_trues)

df = pd.read_csv('Train.csv')

y = df.iloc[:, 6: 6 + 25]
LABELS = y.columns


tdf = pd.read_csv("Test.csv")
i = pkl.load(open(final_pred, "rb"))

for col in range(len(LABELS)):
    tdf[LABELS[col]] = [1 if x[col] > thresholds[col] else 0 for x in i]
    

tdf.drop(columns=['ABSTRACT', 'Computer Science', 'Mathematics', 'Physics', 'Statistics']).to_csv("/tmp/final.csv", index=False)

