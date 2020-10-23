{
"cells": [
{
"metadata": {
"_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
"_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
"trusted": true
},
"cell_type": "code",
"source": "# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only \"../input/\" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session",
"execution_count": 1,
"outputs": [
{
"output_type": "stream",
"text": "/kaggle/input/hacklive-3-guided-hackathon-nlp/SampleSubmission.csv\n/kaggle/input/hacklive-3-guided-hackathon-nlp/Test.csv\n/kaggle/input/hacklive-3-guided-hackathon-nlp/Tags.csv\n/kaggle/input/hacklive-3-guided-hackathon-nlp/Train.csv\n",
"name": "stdout"
}
]
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "# !wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz\n",
"execution_count": 2,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "# # !mkdir scibert_scivocab_uncased\n# !tar -xvf ./scibert_scivocab_uncased.tar.gz\n# ! mv scibert_scivocab_uncased/bert_config.json scibert_scivocab_uncased/config.json\n# ! mv scibert_scivocab_uncased/bert_model.ckpt.data-00000-of-00001 scibert_scivocab_uncased/tf_model.h5\n# !ls scibert_scivocab_uncased\n",
"execution_count": 3,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "!pip install -q transformers==2.11.0\n!wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz\n!tar -xvf ./scibert_scivocab_uncased.tar.gz\nos.environ[\"WANDB_API_KEY\"] = \"0\" ## to silence warning\n!transformers-cli convert --model_type bert \\\n  --tf_checkpoint './scibert_scivocab_uncased/bert_model.ckpt' \\\n  --config './scibert_scivocab_uncased/bert_config.json' \\\n  --pytorch_dump_output './scibert_scivocab_uncased/pytorch_model.bin'",
"execution_count": 4,
"outputs": [
{
"output_type": "stream",
"text": "\u001b[31mERROR: After October 2020 you may experience errors when installing or updating packages. This is because pip will change the way that it resolves dependency conflicts.\n\nWe recommend you use --use-feature=2020-resolver to test your packages with the new resolver before it becomes the default.\n\nallennlp 1.1.0 requires transformers<3.1,>=3.0, but you'll have transformers 2.11.0 which is incompatible.\u001b[0m\n--2020-10-22 22:27:28--  https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz\nResolving s3-us-west-2.amazonaws.com (s3-us-west-2.amazonaws.com)... 52.218.205.56\nConnecting to s3-us-west-2.amazonaws.com (s3-us-west-2.amazonaws.com)|52.218.205.56|:443... connected.\nHTTP request sent, awaiting response... 200 OK\nLength: 1216161420 (1.1G) [application/x-tar]\nSaving to: ‘scibert_scivocab_uncased.tar.gz’\n\nscibert_scivocab_un 100%[===================>]   1.13G  50.6MB/s    in 20s     \n\n2020-10-22 22:27:49 (57.6 MB/s) - ‘scibert_scivocab_uncased.tar.gz’ saved [1216161420/1216161420]\n\nscibert_scivocab_uncased/\nscibert_scivocab_uncased/bert_model.ckpt.data-00000-of-00001\nscibert_scivocab_uncased/bert_model.ckpt.index\nscibert_scivocab_uncased/vocab.txt\nscibert_scivocab_uncased/bert_model.ckpt.meta\nscibert_scivocab_uncased/bert_config.json\n2020-10-22 22:28:11.681048: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.2\nBuilding PyTorch model from configuration: BertConfig {\n  \"attention_probs_dropout_prob\": 0.1,\n  \"hidden_act\": \"gelu\",\n  \"hidden_dropout_prob\": 0.1,\n  \"hidden_size\": 768,\n  \"initializer_range\": 0.02,\n  \"intermediate_size\": 3072,\n  \"layer_norm_eps\": 1e-12,\n  \"max_position_embeddings\": 512,\n  \"model_type\": \"bert\",\n  \"num_attention_heads\": 12,\n  \"num_hidden_layers\": 12,\n  \"pad_token_id\": 0,\n  \"type_vocab_size\": 2,\n  \"vocab_size\": 31090\n}\n\nSave PyTorch model to ./scibert_scivocab_uncased/pytorch_model.bin\n",
"name": "stdout"
}
]
},
{
"metadata": {
"_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a",
"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0",
"trusted": true
},
"cell_type": "code",
"source": "\nimport tensorflow as tf\nimport transformers\nfrom sklearn.model_selection import train_test_split\nfrom tensorflow.keras import backend as K\nimport pickle as pkl\nimport gc\nimport logging\nimport warnings\nimport time\nimport sys",
"execution_count": 5,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "bert_model_name = './scibert_scivocab_uncased'\n\nconfig = transformers.BertConfig.from_json_file('./scibert_scivocab_uncased/bert_config.json')\n\nwarnings.filterwarnings(\"ignore\")\nlogging.basicConfig(level=logging.ERROR)\n\ngpus = tf.config.experimental.list_physical_devices('GPU')\ntf.config.experimental.set_memory_growth(gpus[0], True)\n",
"execution_count": 6,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "\nmax_length = 512\nbatch_size = 32\nepochs = 4\ndf = pd.read_csv('/kaggle/input/hacklive-3-guided-hackathon-nlp/Train.csv')\n\nX = df.iloc[:, 1: 6]\ny = df.iloc[:, 6: 6 + 25]\n\nLABELS = y.columns\n\n\ntrain_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=True, test_size=.2)\ntrain_X.shape, train_y.shape, test_X.shape, test_y.shape\n\n\ndel train_X, train_y\n\n",
"execution_count": 7,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "\n\nX = X.reset_index(drop=True)\ny = y.reset_index(drop=True)\ntest_y = test_y.reset_index(drop=True)\ntest_X = test_X.reset_index(drop=True)\n\n\ntest_y = test_y.values\ny = y.values\n\n",
"execution_count": 8,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "\n\nclass BertSemanticDataGenerator(tf.keras.utils.Sequence):\n\n    def __init__(\n        self,\n        sentences,\n        depts,\n        labels,\n        batch_size=batch_size,\n        shuffle=True,\n        include_targets=True,\n    ):\n        self.sentences = sentences\n        self.depts = depts\n        self.labels = labels\n        self.shuffle = shuffle\n        self.batch_size = batch_size\n        self.include_targets = include_targets\n#         self.tokenizer = transformers.BertTokenizer.from_pretrained(\n#             \"bert-base-uncased\", do_lower_case=True, max_length=2048\n#             )\n        self.tokenizer = transformers.BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)\n        self.indexes = np.arange(len(self.sentences))\n        self.on_epoch_end()\n\n    def __len__(self):\n        return len(self.sentences) // self.batch_size\n\n    def __getitem__(self, idx):\n        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]\n        sentences = self.sentences[indexes]\n\n        encoded = self.tokenizer.batch_encode_plus(\n            sentences.tolist(),\n            add_special_tokens=True,\n            max_length=max_length,\n            return_attention_mask=True,\n            return_token_type_ids=True,\n            pad_to_max_length=True,\n            return_tensors=\"tf\",\n        )\n\n\n        input_ids = np.array(encoded[\"input_ids\"], dtype=\"int32\")\n\n        dept_ids = np.array(self.depts[indexes], dtype=\"int32\")\n        attention_masks = np.array(encoded[\"attention_mask\"], dtype=\"int32\")\n        token_type_ids = np.array(encoded[\"token_type_ids\"], dtype=\"int32\")\n\n        if self.include_targets:\n            labels = np.array(self.labels[indexes], dtype=\"int32\")\n            return [input_ids, dept_ids, attention_masks, token_type_ids], labels\n        else:\n            return [input_ids, dept_ids, attention_masks, token_type_ids]\n\n    def on_epoch_end(self):\n        if self.shuffle:\n            np.random.RandomState(42).shuffle(self.indexes)\n",
"execution_count": 9,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "\ndef recall_m(y_true, y_pred):\n    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))\n    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))\n    recall = true_positives / (possible_positives + K.epsilon())\n    return recall\n\ndef precision_m(y_true, y_pred):\n    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))\n    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))\n    precision = true_positives / (predicted_positives + K.epsilon())\n    return precision\n\ndef f1_m(y_true, y_pred):\n    precision = precision_m(y_true, y_pred)\n    recall = recall_m(y_true, y_pred)\n    return 2*((precision*recall)/(precision+recall+K.epsilon()))\n\ndef macro_double_soft_f1(y, y_hat):\n    \n    y = tf.cast(y, tf.float32)\n    y_hat = tf.cast(y_hat, tf.float32)\n    tp = tf.reduce_sum(y_hat * y, axis=0)\n    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)\n    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)\n    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)\n    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)\n    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)\n    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1\n    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0\n    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0\n    macro_cost = tf.reduce_mean(cost) # average on all labels\n    return macro_cost\n\n",
"execution_count": 10,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "\ninput_ids = tf.keras.layers.Input(\n    shape=(max_length,), dtype=tf.int32, name=\"input_ids\"\n)\n\ninput_departments = tf.keras.Input(\n    shape = (4,), dtype=tf.float32, name='input_depts'\n)\n\n# Attention masks indicates to the model which tokens should be attended to.\nattention_masks = tf.keras.layers.Input(\n    shape=(max_length,), dtype=tf.int32, name=\"attention_masks\"\n)\n\n# Token type ids are binary masks identifying different sequences in the model.\ntoken_type_ids = tf.keras.layers.Input(\n    shape=(max_length,), dtype=tf.int32, name=\"token_type_ids\"\n)\n\n# Loading pretrained BERT model.\n# bert_model = transformers.TFBertModel.from_pretrained(\"bert-base-uncased\")\nbert_model = transformers.TFBertModel.from_pretrained(bert_model_name, from_pt=True, config = config)\n# Freeze the BERT model to reuse the pretrained features without modifying them.\nbert_model.trainable = False\n\nsequence_output, pooled_output = bert_model(\n    input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids\n)\n\n# Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.\nbi_lstm = tf.keras.layers.Bidirectional(\n    tf.keras.layers.LSTM(64, return_sequences=True)\n)(sequence_output)\n\n# Applying hybrid pooling approach to bi_lstm sequence output.\navg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)\nmax_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)\nconcat = tf.keras.layers.concatenate([avg_pool, max_pool, input_departments])\ndropout = tf.keras.layers.Dropout(0.3)(concat)\noutput = tf.keras.layers.Dense(25, activation=\"softmax\")(dropout)\nmodel = tf.keras.models.Model(\n    inputs=[input_ids, input_departments, attention_masks, token_type_ids], outputs=output\n)\n\nmodel.compile(\n    optimizer=tf.keras.optimizers.Adam(),\n    loss=\"binary_crossentropy\",\n    metrics=[\"acc\", f1_m],\n)\n\nmodel.summary()\n",
"execution_count": 11,
"outputs": [
{
"output_type": "stream",
"text": "Model: \"functional_1\"\n__________________________________________________________________________________________________\nLayer (type)                    Output Shape         Param #     Connected to                     \n==================================================================================================\ninput_ids (InputLayer)          [(None, 512)]        0                                            \n__________________________________________________________________________________________________\nattention_masks (InputLayer)    [(None, 512)]        0                                            \n__________________________________________________________________________________________________\ntoken_type_ids (InputLayer)     [(None, 512)]        0                                            \n__________________________________________________________________________________________________\ntf_bert_model (TFBertModel)     ((None, 512, 768), ( 109918464   input_ids[0][0]                  \n                                                                 attention_masks[0][0]            \n                                                                 token_type_ids[0][0]             \n__________________________________________________________________________________________________\nbidirectional (Bidirectional)   (None, 512, 128)     426496      tf_bert_model[0][0]              \n__________________________________________________________________________________________________\nglobal_average_pooling1d (Globa (None, 128)          0           bidirectional[0][0]              \n__________________________________________________________________________________________________\nglobal_max_pooling1d (GlobalMax (None, 128)          0           bidirectional[0][0]              \n__________________________________________________________________________________________________\ninput_depts (InputLayer)        [(None, 4)]          0                                            \n__________________________________________________________________________________________________\nconcatenate (Concatenate)       (None, 260)          0           global_average_pooling1d[0][0]   \n                                                                 global_max_pooling1d[0][0]       \n                                                                 input_depts[0][0]                \n__________________________________________________________________________________________________\ndropout_37 (Dropout)            (None, 260)          0           concatenate[0][0]                \n__________________________________________________________________________________________________\ndense (Dense)                   (None, 25)           6525        dropout_37[0][0]                 \n==================================================================================================\nTotal params: 110,351,485\nTrainable params: 433,021\nNon-trainable params: 109,918,464\n__________________________________________________________________________________________________\n",
"name": "stdout"
}
]
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "# debug",
"execution_count": 12,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "val_data = BertSemanticDataGenerator(\n    test_X['ABSTRACT'].values.astype(\"str\"),\n    test_X.iloc[:,1:6].values.astype(\"int32\"),\n    test_y,\n    batch_size=batch_size,\n    shuffle=False,\n)\n\n\ndata = BertSemanticDataGenerator(\n    X['ABSTRACT'].values.astype(\"str\"),\n    X.iloc[:,1:6].values.astype(\"int32\"),\n    y,\n    batch_size=batch_size,\n    shuffle=True,\n)\n\n\n\n\nprint(\"training model\")\nhistory = model.fit(\n    data,\n    validation_data=val_data,\n    shuffle=True,\n    epochs=epochs,\n)\n\nK.clear_session()\ngc.collect()\nprint(\"sleeping\")\n# time.sleep(60)\n",
"execution_count": 13,
"outputs": [
{
"output_type": "stream",
"text": "training model\nEpoch 1/4\n437/437 [==============================] - 425s 972ms/step - loss: 0.1054 - acc: 0.5137 - f1_m: 0.4209 - val_loss: 0.0756 - val_acc: 0.6365 - val_f1_m: 0.6156\nEpoch 2/4\n437/437 [==============================] - 421s 962ms/step - loss: 0.0786 - acc: 0.6296 - f1_m: 0.6117 - val_loss: 0.0694 - val_acc: 0.6609 - val_f1_m: 0.6665\nEpoch 3/4\n437/437 [==============================] - 420s 961ms/step - loss: 0.0720 - acc: 0.6544 - f1_m: 0.6534 - val_loss: 0.0632 - val_acc: 0.6835 - val_f1_m: 0.6959\nEpoch 4/4\n437/437 [==============================] - 421s 963ms/step - loss: 0.0689 - acc: 0.6708 - f1_m: 0.6757 - val_loss: 0.0597 - val_acc: 0.7119 - val_f1_m: 0.7063\nsleeping\n",
"name": "stdout"
}
]
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "\nepochs = 1\nbatch_size = 4\nval_data = BertSemanticDataGenerator(\n    test_X['ABSTRACT'].values.astype(\"str\"),\n    test_X.iloc[:,1:6].values.astype(\"int32\"),\n    test_y,\n    batch_size=batch_size,\n    shuffle=False,\n)\n\n\ndata = BertSemanticDataGenerator(\n    X['ABSTRACT'].values.astype(\"str\"),\n    X.iloc[:,1:6].values.astype(\"int32\"),\n    y,\n    batch_size=batch_size,\n    shuffle=True,\n)\n\nprint(\"training trainable bert model\")\n\n# history = model.fit(\n#     data,\n#     validation_data=val_data,\n#     epochs=epochs,\n# )\n\n\nval_data = BertSemanticDataGenerator(\n    test_X['ABSTRACT'].values.astype(\"str\"),\n    test_X.iloc[:,1:6].values.astype(\"int32\"),\n    test_y,\n    batch_size=1,\n    shuffle=False,\n)\n\n\nprint(\"predicting validation data\")\npreds = model.predict_generator(val_data, verbose=1, use_multiprocessing=True)\n\n\n\n\nprint(\"evaluating validation data\")\npkl.dump(test_y, open(\"val_original.pkl\", \"wb\"))\npkl.dump(preds, open(\"val_preds.pkl\", \"wb\"))\nmodel.evaluate(val_data, verbose=1)\n\n\ntdf = pd.read_csv(\"/kaggle/input/hacklive-3-guided-hackathon-nlp/Test.csv\")\n\n\nval_data = BertSemanticDataGenerator(\n    tdf['ABSTRACT'].values.astype(\"str\"),\n    tdf.iloc[:,2:6].values.astype(\"int32\"),\n    batch_size=2,\n    labels=None,\n    shuffle=False,\n    include_targets=False\n)\n\n\nprint(\"predicting test data\")\npreds = model.predict_generator(val_data, verbose=1, use_multiprocessing=True)\n\n\nprint(\"final predictions\")\nprint(preds)\n\npkl.dump(preds, open('pred_proba.pkl', 'wb'))\n\n",
"execution_count": 14,
"outputs": [
{
"output_type": "stream",
"text": "training trainable bert model\npredicting validation data\n2801/2801 [==============================] - 182s 65ms/step\nevaluating validation data\n2801/2801 [==============================] - 174s 62ms/step - loss: 0.0596 - acc: 0.7126 - f1_m: 0.6922\npredicting test data\n3001/3001 [==============================] - 266s 88ms/step\nfinal predictions\n[[3.9067065e-05 3.8781126e-03 6.9113627e-02 ... 5.9023838e-05\n  1.4368304e-04 3.2629818e-04]\n [1.3737562e-04 7.1667039e-01 5.3524628e-02 ... 4.6344247e-04\n  4.1122467e-04 9.7100958e-03]\n [6.3911434e-03 3.8423022e-04 1.8464511e-05 ... 1.3578331e-04\n  2.0584227e-04 2.2711964e-04]\n ...\n [8.9671878e-05 4.5979302e-03 2.0740601e-01 ... 8.1028018e-05\n  5.9101174e-05 5.9230230e-03]\n [2.7332734e-03 1.9591156e-01 1.2557521e-02 ... 3.3341214e-02\n  4.8647048e-03 4.1278144e-03]\n [4.6785030e-04 6.6201559e-05 4.5231689e-05 ... 4.3099922e-01\n  1.3202563e-01 1.7456539e-05]]\n",
"name": "stdout"
}
]
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "\n\npred = 'val_preds.pkl'\ny = 'val_original.pkl'\nfinal_pred = 'pred_proba.pkl'\n\n",
"execution_count": 15,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "\npred = pkl.load(open(pred, 'rb'))\n\ny = pkl.load(open(y, 'rb'))\ny = y[0:0 + pred.shape[0]]\n\nthresholds = [0] * 25\ncounts = [0] * 25\ny_trues = [0] * 25\n\ndef get_current_loss(p_col, true_col, th):\n    p_col = sum(p_col > th) / len(p_col)\n    true_col = sum(true_col) / len(true_col)\n    return abs(p_col - true_col)\n\n\nfor col in range(25):\n    current_loss = sys.maxsize\n    temp_loss = sys.maxsize\n    for threshold in np.linspace(0, 1, 80, endpoint=False):\n        pred_col = np.take(pred, col, axis=1)\n        y_col = np.take(y, col, axis=1)\n        temp_loss = get_current_loss(pred_col, y_col, threshold)\n        if  temp_loss < current_loss:\n            thresholds[col] = threshold\n            current_loss = temp_loss\n            counts[col] = sum(pred_col > threshold) / len(pred_col)\n            y_trues[col] = sum(y_col) / len(y_col)\n            # print(col, current_loss, thresholds[col], counts[col], y_trues[col])\n\n\nprint(thresholds)\nprint(counts)\nprint(y_trues)\n",
"execution_count": 16,
"outputs": [
{
"output_type": "stream",
"text": "[0.2, 0.2, 0.2, 0.5, 0.25, 0.25, 0.15000000000000002, 0.15000000000000002, 0.25, 0.25, 0.55, 0.15000000000000002, 0.1, 0.25, 0.35000000000000003, 0.25, 0.4, 0.15000000000000002, 0.15000000000000002, 0.2, 0.2, 0.15000000000000002, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004]\n[0.04569796501249554, 0.05141021063905748, 0.09710817565155301, 0.0417707961442342, 0.04891110317743663, 0.06283470189218136, 0.04569796501249554, 0.034273473759371655, 0.039985719385933594, 0.033916458407711535, 0.021420921099607283, 0.021777936451267403, 0.04069975008925384, 0.282399143163156, 0.048197072474116386, 0.043555872902534806, 0.027133166726169226, 0.04962513388075687, 0.026062120671188863, 0.06890396287040343, 0.04212781149589433, 0.04641199571581578, 0.0567654409139593, 0.04105676544091396, 0.03605855051767226]\n[0.04426990360585505, 0.047126026419136026, 0.10067832916815424, 0.0417707961442342, 0.04998214923241699, 0.06426276329882184, 0.047126026419136026, 0.033916458407711535, 0.039271688682613354, 0.035344519814352014, 0.021420921099607283, 0.02320599785790789, 0.039271688682613354, 0.2788289896465548, 0.047126026419136026, 0.039985719385933594, 0.026776151374509102, 0.04998214923241699, 0.027490182077829346, 0.07104605498036416, 0.04391288825419493, 0.043555872902534806, 0.057836486968939664, 0.04105676544091396, 0.03855765797929311]\n",
"name": "stdout"
}
]
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "\ndf = pd.read_csv('/kaggle/input/hacklive-3-guided-hackathon-nlp/Train.csv')\n\ny = df.iloc[:, 6: 6 + 25]\nLABELS = y.columns\n\n\ntdf = pd.read_csv(\"/kaggle/input/hacklive-3-guided-hackathon-nlp/Test.csv\")\ni = pkl.load(open(final_pred, \"rb\"))\n\nfor col in range(len(LABELS)):\n    tdf[LABELS[col]] = [1 if x[col] > thresholds[col] else 0 for x in i]\n    \n\ntdf.drop(columns=['ABSTRACT', 'Computer Science', 'Mathematics', 'Physics', 'Statistics']).to_csv(\"final.csv\", index=False)",
"execution_count": 17,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "from IPython.display import FileLink\nfilename = \"final.csv\"\nFileLink(filename)",
"execution_count": 18,
"outputs": [
{
"output_type": "execute_result",
"execution_count": 18,
"data": {
"text/plain": "/kaggle/working/final.csv",
"text/html": "<a href='final.csv' target='_blank'>final.csv</a><br>"
},
"metadata": {}
}
]
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "\npred = 'val_preds.pkl'\ny = 'val_original.pkl'\nfinal_pred = 'pred_proba.pkl'",
"execution_count": 19,
"outputs": []
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "FileLink(pred)",
"execution_count": 20,
"outputs": [
{
"output_type": "execute_result",
"execution_count": 20,
"data": {
"text/plain": "/kaggle/working/val_preds.pkl",
"text/html": "<a href='val_preds.pkl' target='_blank'>val_preds.pkl</a><br>"
},
"metadata": {}
}
]
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "FileLink(y)",
"execution_count": 21,
"outputs": [
{
"output_type": "execute_result",
"execution_count": 21,
"data": {
"text/plain": "/kaggle/working/val_original.pkl",
"text/html": "<a href='val_original.pkl' target='_blank'>val_original.pkl</a><br>"
},
"metadata": {}
}
]
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "FileLink(final_pred)",
"execution_count": 22,
"outputs": [
{
"output_type": "execute_result",
"execution_count": 22,
"data": {
"text/plain": "/kaggle/working/pred_proba.pkl",
"text/html": "<a href='pred_proba.pkl' target='_blank'>pred_proba.pkl</a><br>"
},
"metadata": {}
}
]
},
{
"metadata": {
"trusted": true
},
"cell_type": "code",
"source": "",
"execution_count": null,
"outputs": []
}
],
"metadata": {
"kernelspec": {
"name": "python3",
"display_name": "Python 3",
"language": "python"
},
"language_info": {
"name": "python",
"version": "3.7.6",
"mimetype": "text/x-python",
"codemirror_mode": {
"name": "ipython",
"version": 3
},
"pygments_lexer": "ipython3",
"nbconvert_exporter": "python",
"file_extension": ".py"
}
},
"nbformat": 4,
"nbformat_minor": 4
}
