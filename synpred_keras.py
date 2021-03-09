"""
Script to deploy tensorflow for Gridsearch
tensorflow version 2
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.losses import BinaryCrossentropy
import pandas as pd
from sklearn import preprocessing
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, \
                            recall_score, precision_score, f1_score
from synpred_variables import RANDOM_STATE, DROPPABLE_COLUMNS, \
                                CELL_TYPES
import sys
import random
import numpy as np
import ast
from synpred_support_functions import model_evaluation
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.compat.v1.set_random_seed(RANDOM_STATE)
c_type = 'all'

__author__ = "A.J.Preto & Pedro Matos-Filipe"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

class neural_network_class:

    """
    Standard neural network class for iterative deployment
    """
    def __init__(self, input_architecture,  \
                        input_features, \
                        activation_function = "relu", \
                        add_dropout = True, \
                        dropout_rate = 0.5):
        self.model = Sequential()
        self.model.add(Dense(input_architecture[0], input_dim = input_features, \
                                    activation = activation_function, \
                                    kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                    bias_regularizer = regularizers.l2(1e-4), \
                                    activity_regularizer = regularizers.l2(1e-5)))
        for hidden_layer in input_architecture[1:]:
            if add_dropout == True:
                self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(hidden_layer, activation = activation_function, \
                                    kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                    bias_regularizer = regularizers.l2(1e-4), \
                                    activity_regularizer = regularizers.l2(1e-5)))
            
        self.model.add(Dense(1, activation='sigmoid'))

def prepare_dataset(file = '', method = 'Full-agreement', \
            c_type='all', sample_fraction = 0.1, \
            sample_fraction_mode = False):
    """
    Prepare the dataset to be trained
    """
    input_dataframe = pd.read_csv(file)
    if sample_fraction_mode == True:
        input_dataframe = input_dataframe.sample(frac = sample_fraction)

    if c_type != 'all':
        if type(c_type) is list:
            exclusion = CELL_TYPES[c_type[0]]
            for i in c_type[1:]:
                exclusion += CELL_TYPES[i]
        else:
            exclusion = CELL_TYPES[c_type]

    if c_type != 'all':
        input_dataframe = input_dataframe.loc[input_dataframe['cell'].isin(exclusion)]

    names = input_dataframe[['cell', 'drug1', 'drug2']]
    features = input_dataframe

    for column in DROPPABLE_COLUMNS:
        if column != method:
            features = features.drop([column], axis = 1)
        else:
            continue

    target = features.pop(method)
    return names, features, target

def model_training(dataset, method, input_model, \
                    c_type='all', outp = None, save_model = True):

    """
    Train the dataset
    """
    names, features, target = prepare_dataset(file = dataset, method = method, \
                                                c_type = c_type, sample_fraction = 0.1, \
                                                sample_fraction_mode = True)
    history = input_model.model.fit(x = features, y = target, epochs = 250, validation_split = 0.10)
    if save_model == True:
        input_model.model.save('./saved_model/{}_{}'.format(outp, method))

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_csv('train_log/training_metrics_{}_{}.csv'.format(outp, method))
    predicted_class = input_model.model.predict(features)
    return input_model, target, predicted_class

input_mode = sys.argv[3]
method = "Full-agreement"
train = "./datasets/train_" + input_mode + ".csv"
test = "./datasets/test_" + input_mode + ".csv"
outp = sys.argv[-1]
input_architecture = [int(x) for x in ast.literal_eval(sys.argv[1])]

if input_mode.split("_")[0] == "PCA":
    input_features = 1347
elif input_mode.split("_")[0] == "autoencoder":
    input_features = 4229

raw_model =  neural_network_class(input_architecture, input_features, dropout_rate = float(sys.argv[2]))
optimizer = tf.keras.optimizers.Adam(0.0001)
raw_model.model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
classification_model, train_class, train_predictions = model_training(train, \
                                                    method, raw_model, c_type = c_type, \
                                                    outp = outp, save_model = False)

names, test, target = prepare_dataset(file = test, method = method, c_type = c_type)
test_predictions = [int(np.round(x)) for x in classification_model.model.predict(test)]
train_predictions = [int(np.round(x)) for x in train_predictions]
model_evaluation(target, test_predictions, subset_type = outp + "_test")
model_evaluation(train_class, train_predictions, subset_type = outp + "_train")

sys.exit()

names['target'] = list(target)
names['prediction'] = list(test_predictions)
classification_model.model.summary()

names.to_csv('predictions_{}_{}.csv'.format(outp, method))