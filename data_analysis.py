#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk
tqdm.pandas()


# In[ ]:


df = pd.read_csv('train.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


# df['Computer Science'].value_counts(), df['Physics'].value_counts(), df['Mathematics'].value_counts()


# In[ ]:


# fig, ax = plt.subplots()
# df.ABSTRACT.str.len().value_counts().plot(ax=ax, kind='bar')

# df.TITLE.str.len().value_counts().plot(ax=ax, kind='bar')


# In[ ]:


df.ABSTRACT.str.len().describe()


# In[ ]:


df.TITLE.str.len().describe()


# In[ ]:


np.max(df.TITLE.str.len()), np.min(df.TITLE.str.len())


# In[ ]:


np.max(df.ABSTRACT.str.len()), np.min(df.ABSTRACT.str.len())


# In[ ]:


df.ABSTRACT.str.len().value_counts().sort_index()


# In[ ]:


df.TITLE.str.len().value_counts().sort_index()


# In[ ]:



def data_text_preprocess(total_text):
    # Remove int values from text data as that might not be imp
    if type(total_text) is not int:
        string = ""
        # replacing all special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # replacing multiple spaces with single space
        total_text = re.sub('\s+',' ', str(total_text))
        # bring whole text to same lower-case scale.
#         total_text = total_text.lower()
#         total_text = nltk.pos_tag(total_text.split())
#         total_text = " ".join([i[0] for i in total_text if "NN" in i[1]])
        for word in total_text.split():
            #if not word in stop_words:
            string += word + " "
        
        return string
    return ""
stop_words = set(stopwords.words('english'))

df['ABSTRACT'] = df['ABSTRACT'].progress_apply(lambda x: data_text_preprocess(x))
df['TITLE'] = df['TITLE'].progress_apply(lambda x: data_text_preprocess(x))


# In[ ]:


df.head()


# In[ ]:


df.ABSTRACT.str.len().value_counts().sort_index()


# In[ ]:


df.TITLE.str.len().value_counts().sort_index()


# In[ ]:


df.ABSTRACT.str.len().describe()


# In[ ]:


df.TITLE.str.len().describe()


# In[ ]:


np.max(df.ABSTRACT.str.len()), np.min(df.ABSTRACT.str.len())


# In[ ]:


np.max(df.TITLE.str.len()), np.min(df.TITLE.str.len())


# In[ ]:


df.set_index("ID", inplace=True)


# In[ ]:


df.head()


# In[ ]:


# df.to_csv("cleaned_data.csv")
def add(abstract, title, max_len=None):
    new_abstract, new_title = [], []
    abstract, title = abstract.split(), title.split()
    if not max_len:
        return ' '.join(abstract + title)
    for i in range(max_len):
        #print(abstract[i])
        try: new_abstract.append(abstract[i])
        except: new_abstract.append('__PAD__')
        try: new_title.append(title[i])
        except: new_title.append('__PAD__')
    return ' '.join(new_title + new_abstract)
columns = ['Computer Science', 'Physics', 'Mathematics',
       'Statistics', 'Quantitative Biology', 'Quantitative Finance']
def get_label(row):
    label = [0, 0, 0, 0, 0, 0]
    for col in range(len(columns)):
        if row[columns[col]] == 1:
            label[col] = 1
    return np.array(label)


# In[ ]:



df['text'] = df.progress_apply(lambda row: add(row['ABSTRACT'], row['TITLE']), axis=1)

df['labels'] = df.progress_apply(lambda row: get_label(row), axis=1)


# In[ ]:



df.text.str.len().describe()


# In[ ]:


df.text.str.len().value_counts().sort_index()


# In[ ]:


df.drop(df.columns.difference(['text','labels']), 1, inplace=True)


# In[ ]:


from simpletransformers.classification  import MultiLabelClassificationModel
import logging


# In[ ]:


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
# train_data = [['Example sentence 1 for multilabel classification.', [1, 1, 1, 1, 0, 1]], ['This is another example sentence. ', [0, 1, 1, 0, 0, 0]]]
# train_df = pd.DataFrame(train_data, columns=['text', 'labels'])


# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel('roberta', 'roberta-base-openai-detector', num_labels=6, args={'no_cache': True, 'train_batch_size': 2, 'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 5}, use_cuda=True)
# You can set class weights by using the optional weight argument
# print(train_df.head())

# Train the model
model.train_model(df)

# Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)
# print(result)
# print(model_outputs)


# In[ ]:





# In[ ]:



tdf = pd.read_csv('test.csv')



tdf['ABSTRACT'] = tdf['ABSTRACT'].progress_apply(lambda x: data_text_preprocess(x))
tdf['TITLE'] = tdf['TITLE'].progress_apply(lambda x: data_text_preprocess(x))

tdf['text'] = tdf.progress_apply(lambda row: add(row['ABSTRACT'], row['TITLE']), axis=1)


# In[ ]:


tdf.iloc[0]['text']


# In[ ]:

def assign_values_to_cols(val, idx):
    return val[idx]

for col in tqdm(range(len(columns))):
    tdf[columns[col]] = tdf['text'].apply(lambda x: model.predict([x])[1][0][col])
# predictions, raw_outputs = model.predict([tdf.iloc[0]['text']])
# print(predictions)
# print(raw_outputs)
tdf.to_csv('test_temp.csv', index=False)
    




# In[ ]:




