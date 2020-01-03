# This file implements a flask server to do inferences.

import os
import json
import pickle
import sys
import signal
import traceback

import flask

import numpy as np
import pandas as pd
import processing

from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer
from keras.models import load_model
from keras.utils import CustomObjectScope
from tensorflow import matmul

#prefix = '/opt/ml/'
#model_path = os.path.join(prefix, 'model')
attention_dim = processing.attention_dim
model_path = 'models/'

class HierarchicalAttentionNetwork(Layer):
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
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

# A singleton that loads the model the first time it is called, holds it and makes predictions.
class Predictor(object):

    @classmethod
    def get_lr_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.lr_model == None:
            with open(os.path.join(model_path, 'lr_model.pkl'), 'rb') as model:
                cls.lr_model = pickle.load(model)
            with open(os.path.join(model_path, 'tfidf.pkl'), 'rb') as model:
                cls.tfidf = pickle.load(model)
            with open(os.path.join(model_path, 'counter.pkl'), 'rb') as model:
                cls.counter = pickle.load(model)
        return cls.lr_model, cls.tfidf, cls.counter

    @classmethod
    def get_keras_model(cls):
        if cls.keras_model == None:
            model = os.path.join(model_path, 'keras_model.hdf5')
            with CustomObjectScope({'HierarchicalAttentionNetwork': HierarchicalAttentionNetwork}):
                cls.keras_model = load_model(model)
        with open(os.path.join(model_path, 'word_index.pkl'), 'rb') as model:
                cls.word_index = pickle.load(model)
        return cls.keras_model, cls.word_index

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        
        article = processing.reformat(input)
        if not article:
            return -1

        keras_model, word_index = cls.get_keras_model()
        lr_model, tfidf, counter = cls.get_lr_model()

        keras_art = processing.create_data_matrix([article], word_index=word_index)
        keras_output = keras_model.predict(x=keras_art, steps=1)[0][1]

        lr_art = [processing.recombine(article)]
        transformed = counter.transform(lr_art)
        lr_output = lr_model.predict_proba(tfidf.transform(transformed))[0][1]

        print('keras output = ', keras_output, '\nlr output = ', lr_output)
        return (keras_output + lr_output) / 2

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Pass health check if model artifacts load and a correct prediction is made.
    """
    health = Predictor.predict('Too short') == -1
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def score():
    """Do an inference on a single block of text passed in through a web form.
    """
    data = None
    try:
        if flask.request.data:
            data = flask.request.data.decode('utf-8')
        else:
            data = flask.request.form['article']

        print('Invoked with article containing {} characters'.format(len(data)))

        # Do the prediction
        result = Predictor.predict(data)
        result = f'Our model gives this article a reliability rating of {round(result * 100, 1)} percent.'
        status = 200

    except Exception as e:
            trc = traceback.format_exc()
            print('Exception during processing: ' + str(e) + '\n' + trc)
            result = 'This predictor only supports single blocks of text'
            status = 415

    return flask.Response(response=result, status=status, mimetype='text/plain')