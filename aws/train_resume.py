#!/usr/bin/env python

import numpy as np
import pandas as pd
import boto3
import pickle
import gc
import os
import sys
import traceback

import utils.config as config

from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import CustomObjectScope

data_dir = '/opt/ml/input/data/train/'
model_dir = '/opt/ml/model/'
checkpoints_dir = '/opt/ml/checkpoints/'
output_dir = '/opt/ml/output/'

X_train_file, X_val_file, y_train_file, y_val_file = \
    data_dir + 'X_train.pkl', data_dir + 'X_val.pkl', data_dir + 'y_train.pkl', data_dir + 'y_val.pkl'
with open(X_train_file, 'rb') as infile:
    X_train = pickle.load(infile)
with open(X_val_file, 'rb') as infile:
    X_val = pickle.load(infile)
with open(y_train_file, 'rb') as infile:
    y_train = pickle.load(infile)
with open(y_val_file, 'rb') as infile:
    y_val = pickle.load(infile)

max_words = config.max_words  # max num words processed for each sentence
max_sentences = config.max_sentences  # max num sentences processed for each article 
max_vocab = config.max_vocab
attention_dim = config.attention_dim
epoch = config.epoch
batch_size = config.batch_size
words_file = data_dir + 'words.pkl'
saved_model_name = 'model.{}.hdf5'.format(epoch - 1)
saved_model = data_dir + saved_model_name

y_train = np.asarray(to_categorical(y_train))
y_val = np.asarray(to_categorical(y_val))

with open(words_file, 'rb') as infile:
    words = pickle.load(infile)
word_index = {}
for ix, (word, _) in enumerate(words.most_common(max_vocab)):
    word_index[word] = ix + 1

def create_data_matrix(data, max_sentences=max_sentences, max_words=max_words, max_vocab=max_vocab,
                      word_index=word_index):
    data_matrix = np.zeros((len(data), max_sentences, max_words), dtype='int32')
    for i, article in enumerate(data):
        for j, sentence in enumerate(article):
            if j == max_sentences:
                break
            k = 0
            for word in sentence:
                if k == max_words:
                    break
                ix = word_index.get(word.lower())
                if ix is not None and ix < max_vocab:
                    data_matrix[i, j, k] = ix
                k = k + 1
    return data_matrix

X_train = create_data_matrix(X_train)
X_val = create_data_matrix(X_val)

from tensorflow import matmul
class HierarchicalAttentionNetwork(Layer):
    ''''''
    def __init__(self, **kwargs):
        self.init_weights = initializers.get('glorot_normal')
        self.init_bias = initializers.get('zeros')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super().__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init_weights((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init_bias((self.attention_dim,)))
        self.u = K.variable(self.init_weights((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):        
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))
        
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

if __name__ == "__main__":
    try:
        with CustomObjectScope({'HierarchicalAttentionNetwork': HierarchicalAttentionNetwork}):
            model = load_model(saved_model)
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                batch_size=batch_size, epochs=1)
        X_train = None
        X_val = None
        y_train = None
        y_val = None
        gc.collect()
        model.save(os.path.join(model_dir, 'model.{}.hdf5'.format(epoch)))
        with open(os.path.join(model_dir, 'history.{}.pkl'.format(epoch)), 'wb') as outfile:
            pickle.dump(hist.history, outfile)
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_dir, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

    sys.exit(0)
