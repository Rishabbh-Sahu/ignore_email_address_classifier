# -*- coding: utf-8 -*-
"""
Created on Sat Apr 03
@author: rishabbh-sahu
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import yaml
import math
import json


from tokenizer import create_bert_tokenizer,tokenize_text
from model import TEXT_MODEL

configuration_file_path = 'config.yaml'

def read_yaml_from_file(filepath):
    with open(filepath) as fp:
        return yaml.load(fp, Loader=yaml.SafeLoader)

config = {}
config.update(read_yaml_from_file('config.yaml'))

# Reading the data file
data = pd.read_csv(config['data_path'])
data.isnull().values.any()
print(f'Data shape : {data.shape}')
print(f'Column names : {data.columns}')

# shuffling the records multiple times
data = data.sample(frac = 1)
data = data.sample(frac = 1)
data = data.sample(frac = 1)
print(f"Labels distribution \n {np.round(data[config['label_field_name']].value_counts()/data.shape[0]*100,2)}")

# Reading the text column
print(f'reading data...')
text_data = []
text_data = list(data[config['text_field_name']])
print(f'Text input sample records {text_data[:5]}')

label_data = data[config['label_field_name']]
config['OUTPUT_CLASSES']=len(set(label_data))
label_data = np.array(list(map(lambda x: 1 if x=="Yes" else 0, label_data)))
print(f'Text input label sample records {label_data[:5]}')

tokenizer = create_bert_tokenizer(config['model_path'])
config['VOCAB_LENGTH'] = len(tokenizer.vocab)

tokenized_text = [tokenize_text(text,tokenizer) for text in text_data]

text_with_len = [[text, label_data[i], len(text)] for i, text in enumerate(tokenized_text)]
random.shuffle(text_with_len)
text_with_len.sort(key=lambda x: x[2])

sorted_text_labels = [(text[0], text[1]) for text in text_with_len]
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_text_labels, output_types=(tf.int32, tf.int32))

BATCH_SIZE = config['BATCH_SIZE']
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
next(iter(batched_dataset))

TOTAL_BATCHES = math.ceil(len(sorted_text_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 10
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)

model = TEXT_MODEL(vocabulary_size=config['VOCAB_LENGTH'],
                   embedding_dimensions=config['EMB_DIM'],
                   cnn_filters=config['CNN_FILTERS'],
                   dnn_units=config['DNN_UNITS'],
                   model_output_classes=config['OUTPUT_CLASSES'],
                   dropout_rate=config['DROPOUT_RATE'])

optimizer = 'adam'
metrics = ["accuracy"]

if config['OUTPUT_CLASSES'] == 2:
    loss = "binary_crossentropy"
else:
    loss = "sparse_categorical_crossentropy"

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.fit(train_data, epochs=config['EPOCHS'],verbose=1)

results = model.evaluate(test_data)

print(f"Saving model and its config here - {os.path.join(config['saved_model_dir_path'],config['model_name'],config['model_version'])}")
model.save(os.path.join(config['saved_model_dir_path'],config['model_name'],config['model_version']))
with open(os.path.join(config['saved_model_dir_path'],config['model_name'],config['model_version'],'config.json'),'w') as fp:
    json.dump(config,fp)