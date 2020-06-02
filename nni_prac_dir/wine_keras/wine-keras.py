import argparse
import logging

import os

import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping,TensorBoard
from keras.layers import Dense, Dropout
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

MODEL_DIR = './model'

def create_wine_model(hyper_params, input_shape=(FEATURE_NUM,), num_classes=NUM_CLASSES):
    '''
    Create model
    '''
    layers = [
        Dense(hyper_params['hidden_size1'], activation='relu', input_shape=input_shape),
        Dropout(rate=hyper_params['drop_out_late1'], seed=7),
        Dense(hyper_params['hidden_size2'], activation='relu'),
        Dropout(rate=hyper_params['drop_out_late2'], seed=7),
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

    es_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
        validation_data=(x_test, y_test), callbacks=[SendMetrics(), TensorBoard(log_dir=TENSORBOARD_DIR), es_callback])

    _, train_acc = model.evaluate(x_test, y_test, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    fpath = 'epochs{e:02d}-train_acc{train_acc:.2f}-test_acc{test_acc:.2f}'.format(e=args.epochs, train_acc=train_acc, test_acc=test_acc)
    model.save_weights(os.path.join(MODEL_DIR, fpath+'.hdf5'))
    json_string = model.to_json()
    open(os.path.join(MODEL_DIR, fpath+'.json'), 'w').write(json_string)

    LOG.debug('Final result is: %d', test_acc)
    nni.report_final_result(test_acc)

def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'hidden_size1': 20,
        'hidden_size2': 25,
        'drop_out_late1': 0.5,
        'drop_out_late2': 0.5,
        'optimizer': 'Adam',
        'learning_rate': 0.001
    }

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--batch_size", type=int, default=256, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=30, help="Train epochs", required=False)

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
