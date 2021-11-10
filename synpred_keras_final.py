#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to deploy tensorflow fully
"""

__author__ = "A.J.Preto & Pedro Matos-Filipe"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

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
import sys
import random
import numpy as np
import ast
import os
from synpred_support_functions import model_evaluation
from synpred_variables import INTERMEDIATE_SEP, DL_SAVED_MODELS, \
                                SYSTEM_SEP, RANDOM_STATE, \
                                DROPPABLE_COLUMNS, DATASET_FOLDER

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.compat.v1.set_random_seed(RANDOM_STATE)

class neural_network:

    """
    Standard neural network class for iterative deployment
    """
    def __init__(self, input_architecture,  \
                        input_features, \
                        activation_function = "relu", \
                        add_dropout = True, \
                        dropout_rate = 0.5, \
                        prediction_mode = "regression"):
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
        if prediction_mode == "classification":
            self.model.add(Dense(1, activation = 'sigmoid'))
        elif prediction_mode == "regression":
            self.model.add(Dense(1, activation = 'linear'))

def prepare_dataset(file = '', method = 'full_agreement', \
            sample_fraction = 0.1, sample_fraction_mode = False, \
            task_type = "regression"):
    """
    Prepare the dataset to be trained
    """
    input_dataframe = pd.read_csv(file)
    if task_type == "classification":
        input_dataframe = input_dataframe.loc[input_dataframe["full_agreement_val"] == 1]
    if sample_fraction_mode == True:
        input_dataframe = input_dataframe.sample(frac = sample_fraction)

    names = input_dataframe[['cell', 'Drug1', 'Drug2']]
    features = input_dataframe

    for column in DROPPABLE_COLUMNS:
        if (column != method) and (column in list(features)):
            features = features.drop([column], axis = 1)
        else:
            continue

    target = features.pop(method)
    return names, features, target

def model_training(dataset, method, input_model, \
                    outp = None, save_model = True, \
                    problem_type = "classification"):

    """
    Train the dataset
    """
    names, features, target = prepare_dataset(file = dataset, method = method, \
                                                sample_fraction_mode = False, \
                                                task_type = problem_type)
    history = input_model.model.fit(x = features, y = target, epochs = 125, validation_split = 0.10)
    if save_model == True:
        input_model.model.save(os.path.join("./saved_model/", outp + INTERMEDIATE_SEP + method + ".h5"))

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_csv('train_log/training_metrics_{}_{}.csv'.format(outp, method))
    predicted_class = input_model.model.predict(features)
    return input_model, target, predicted_class

input_mode = sys.argv[3]
target_col = sys.argv[4]

if target_col == "full_agreement":
    problem_type = "classification"
else:
    problem_type = "regression"

method = target_col
train = DATASET_FOLDER + SYSTEM_SEP + input_mode.split(INTERMEDIATE_SEP)[0] + INTERMEDIATE_SEP + \
            "train" + INTERMEDIATE_SEP + input_mode.split(INTERMEDIATE_SEP)[1] + ".csv"
test = DATASET_FOLDER + SYSTEM_SEP + input_mode.split(INTERMEDIATE_SEP)[0] + INTERMEDIATE_SEP + \
            "test" + INTERMEDIATE_SEP + input_mode.split(INTERMEDIATE_SEP)[1] + ".csv"
outp = sys.argv[-1]
input_architecture = [int(x) for x in ast.literal_eval(sys.argv[1])]

if input_mode.split(INTERMEDIATE_SEP)[0] == "PCA":
    input_features = 1347
elif input_mode.split(INTERMEDIATE_SEP)[0] == "autoencoder":
    input_features = 4229

raw_model =  neural_network(input_architecture, input_features, \
                            dropout_rate = float(sys.argv[2]), prediction_mode = problem_type)
optimizer = tf.keras.optimizers.Adam(0.0001)

if problem_type == "classification":
    
    raw_model.model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    classification_model, train_class, train_predictions = model_training(train, \
                                                        method, raw_model, outp = outp, \
                                                        save_model = True, problem_type = problem_type)

    names, test, target = prepare_dataset(file = test, method = method, task_type = "classification", sample_fraction_mode = False)
    test_predictions = [int(np.round(x)) for x in classification_model.model.predict(test)]
    train_predictions = [int(np.round(x)) for x in train_predictions]
    model_evaluation(target, test_predictions, subset_type = outp + "_test", \
                        task_type = problem_type)
    model_evaluation(train_class, train_predictions, subset_type = outp + "_train", \
                        task_type = problem_type)

elif problem_type == "regression":
    raw_model.model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse','mae'])
    regression_model, train_class, train_predictions = model_training(train, \
                                                        method, raw_model, outp = outp, \
                                                        save_model = True, problem_type = problem_type)

    names, test, target = prepare_dataset(file = test, method = method, task_type = "regression", sample_fraction_mode = False)
    test_predictions = list(regression_model.model.predict(test)[:,0])
    train_predictions = list(train_predictions[:,0])
    model_evaluation(target, test_predictions, subset_type = outp + "_test", \
                        task_type = problem_type)
    model_evaluation(train_class, train_predictions, subset_type = outp + "_train", \
                        task_type = problem_type)