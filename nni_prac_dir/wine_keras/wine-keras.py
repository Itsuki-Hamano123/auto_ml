# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging

import os

import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

import nni

LOG = logging.getLogger('wine_keras')
K.set_image_data_format('channels_last')
TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR']

DATASET_DIR = './big_dataset'
TRAIN_CSV = 'preprocessed_train.csv'
TEST_CSV = 'preprocessed_test.csv'

LABEL_COLUM = 'class'
FEATURE_NUM = 13
NUM_CLASSES = 3



def create_wine_model(hyper_params, input_shape=(FEATURE_NUM,), num_classes=NUM_CLASSES):
    '''
    Create model
    '''
    layers = [
        Dense(20, activation='relu', input_shape=input_shape),
        Dense(25, activation='relu'),
        Dense(num_classes, activation='softmax')
    ]

    model = Sequential(layers)

    if hyper_params['optimizer'] == 'Adam':
        optimizer = keras.optimizers.Adam(lr=hyper_params['learning_rate'])
    else:
        optimizer = keras.optimizers.SGD(lr=hyper_params['learning_rate'], momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model

def load_wine_data():
    '''
    Load wine dataset
    '''
    train_df = pd.read_csv(os.path.join(DATASET_DIR, TRAIN_CSV))
    x_train = train_df.drop(columns=LABEL_COLUM).values
    y_train = keras.utils.to_categorical(train_df[LABEL_COLUM].values, NUM_CLASSES)

    LOG.debug('x_train shape: %s', (x_train.shape,))


    test_df = pd.read_csv(os.path.join(DATASET_DIR, TEST_CSV))
    x_test = test_df.drop(columns=LABEL_COLUM).values
    y_test = keras.utils.to_categorical(test_df[LABEL_COLUM], NUM_CLASSES)

    LOG.debug('x_test shape: %s', (x_test.shape,))

    return x_train, y_train, x_test, y_test

class SendMetrics(keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        LOG.debug(logs)
        nni.report_intermediate_result(logs["val_acc"])

def train(args, params):
    '''
    Train model
    '''
    x_train, y_train, x_test, y_test = load_wine_data()
    model = create_wine_model(params)

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
        validation_data=(x_test, y_test), callbacks=[SendMetrics(), TensorBoard(log_dir=TENSORBOARD_DIR)])

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    LOG.debug('Final result is: %d', acc)
    nni.report_final_result(acc)

def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'optimizer': 'Adam',
        'learning_rate': 0.001
    }

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--batch_size", type=int, default=10, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=10, help="Train epochs", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = generate_default_params()
        PARAMS.update(RECEIVED_PARAMS)
        # train
        train(ARGS, PARAMS)
    except Exception as e:
        LOG.exception(e)
        raise
