#!/usr/bin/env python

import numpy as np
import pandas as pd
import boto3
import pickle
import gc
import sys
import traceback

import utils.config as config
from cyclic.rate_cycler import CyclicLR
from utils.hierarchical import HierarchicalAttentionNetwork

from keras import backend as K
from keras.models import Model
from keras import initializers
from keras.engine.topology import Layer
from keras.layers import Input, Dropout, Dense
from keras.layers import Embedding, GRU, LSTM, Bidirectional, TimeDistributed
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback, LambdaCallback
from keras.optimizers import SGD, Adam

from tensorflow import set_random_seed, matmul

np.random.seed(3)
set_random_seed(24)
data_dir = '/opt/ml/input/data/'
model_dir = '/opt/ml/model'

X_train_file, X_val_file, y_train_file, y_val_file = \
    data_dir + 'X_train.pkl', data_dir + 'X_val.pkl', data_dir + 'y_train.pkl', data_dir + 'y_val.pkl'
with open(X_train_file, 'rb') as infile:
    X_train = pickle.load(infile).sample(800)
train_indices = X_train.index
with open(X_val_file, 'rb') as infile:
    X_val = pickle.load(infile).sample(200)
val_indices = X_val.index
with open(y_train_file, 'rb') as infile:
    y_train = pickle.load(infile)
y_train = y_train.loc[train_indices]
with open(y_val_file, 'rb') as infile:
    y_val = pickle.load(infile)
y_val = y_val.loc[val_indices]

max_words = config.max_words  # max num words processed for each sentence
max_sentences = config.max_sentences  # max num sentences processed for each article 
max_vocab = config.max_vocab
embedding_dim = config.embedding_dim  # size of pretrained word vectors
attention_dim = config.attention_dim  # num units in attention layer
GRU_dim = config.GRU_dim  # num units in GRU layer, but it is bidirectional so outputs double this number
epochs = config.epochs
batch_size = config.batch_size
test_size = config.test_size

vector_file = data_dir + 'glove.6B.200d.txt'

num_samples = X_train.shape[0]
y_train = np.asarray(to_categorical(y_train))
y_val = np.asarray(to_categorical(y_val))

with open('data/words.pkl', 'rb') as infile:
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

def store_embeddings(vector_file=vector_file):
    embeddings = {}
    with open(vector_file) as vectors:
        for line in vectors:
            values = line.split()
            word = values[0]
            weights = np.asarray(values[1:], dtype='float32')
            embeddings[word] = weights
    return embeddings
            
embeddings = store_embeddings()

def create_embedding_matrix(max_vocab=max_vocab, embeddings=embeddings, word_index=word_index,
                            embedding_dim=embedding_dim):
    embedding_matrix = np.zeros((max_vocab + 1, embedding_dim)) # max_vocab + 1 to account for 0 as masking index
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will remain all zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
            
embeddings = create_embedding_matrix()
gc.collect()

def build_model(attention_dim=attention_dim, GRU_dim=GRU_dim, drop=False, drop_pct=None,
                embedding_matrix=embeddings, embedding_dim=embedding_dim, word_index=word_index):
    
    embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                input_length=max_words, trainable=False, mask_zero=True)

    #  Layers for processing words in each sentence with attention; output is encoded sentence vector 
    sentence_input = Input(shape=(max_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    lstm_word = Bidirectional(GRU(GRU_dim, return_sequences=True))(embedded_sequences)
    attn_word = HierarchicalAttentionNetwork(attention_dim)(lstm_word)
    sentence_encoder = Model(sentence_input, attn_word)
    
    #  Layers for processing sentences in each article with attention; output is prediction
    article_input = Input(shape=(max_sentences, max_words), dtype='int32')
    article_encoder = TimeDistributed(sentence_encoder)(article_input)
    lstm_sentence = Bidirectional(GRU(GRU_dim, return_sequences=True))(article_encoder)
    attn_sentence = HierarchicalAttentionNetwork(attention_dim)(lstm_sentence)

    #  The Adam optimizer also will be tried and can take a dropout layer
    if drop:
        drop_sentence = Dropout(drop_pct)(attn_sentence)
        preds = Dense(2, activation='softmax')(drop_sentence)
    else:
        preds = Dense(2, activation='softmax')(attn_sentence)
    
    return Model(article_input, preds)

model_checkpoint = ModelCheckpoint(filepath='opt/ml/model/weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5')
clr = CyclicLR(epochs=epochs, num_samples=num_samples, batch_size=batch_size)


if __name__ == "__main__":

    try:
        model = build_model()
        opt = SGD(momentum=0.9)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                batch_size=batch_size, epochs=epochs, callbacks=[clr, model_checkpoint])

        with open('/opt/ml/model/history.pkl', 'wb') as outfile:
            pickle.dump(model.History.history, outfile)
        model.save('opt/ml/model/model.h5')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open('opt/ml/output/failure', 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

    sys.exit(0)
