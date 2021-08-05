# -*- coding: utf-8 -*-
"""
@author: rishabbh-sahu
"""

import tensorflow as tf
from tensorflow.keras import layers

class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128, #this parameter lead to increase in the size of the model
                 cnn_filters=50, #parallel fields for processing words
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)
        
        '''
        for text classfication, do try the following word embeddings -
            word2vec from jensim - guide of using this can be found here - "https://radimrehurek.com/gensim/models/word2vec.html"
            GloVe from google - download the pre-learned embeddings from here "https://nlp.stanford.edu/projects/glove/"
        '''
        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        #Feature Learning 
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        #Pooling is required to down sample the detection of features in feature maps
        self.pool = layers.GlobalMaxPool1D()

        #Classification neural network
        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        # concatenated = tf.concat([l_1, l_2], axis=-1)     # experiment by enabling/disabling layers and compute accuracy
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output
