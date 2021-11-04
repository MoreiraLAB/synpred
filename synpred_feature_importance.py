#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate feature importance for Sankey plot construction
"""

import pandas as pd
import eli5
import sklearn
from eli5.sklearn import PermutationImportance
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
                                RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from synpred_variables import SYSTEM_SEP, PARAGRAPH_SEP, CSV_SEP, \
                            RANDOM_STATE, EVALUATION_NON_DL_FOLDER, \
                            INTERMEDIATE_SEP, REDEPLOYMENT_FOLDER, DATASETS_DICTIONARY_NO_CONCENTRATION, \
                            ML_GRIDSEARCH_PARAMETERS, METRICS_CLASSIFICATION, METRICS_REGRESSION, \
                            POSSIBLE_TARGETS, FEATURE_IMPORTANCE_FOLDER, CSV_TERMINATION, \
                            DL_GRIDSEARCH_PARAMETERS
import xgboost as xgb
import pickle
from synpred_support_functions import prepare_dataset
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.losses import BinaryCrossentropy
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
from sklearn.metrics import mean_squared_error, accuracy_score, make_scorer
__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.compat.v1.set_random_seed(RANDOM_STATE)

FEATURE_IMPORTANCE_START = "feature_importance"

def tailored_accuracy(y_true, y_predicted):

    """
    Calculate accuracy for keras model
    """
    numerator = 0
    denominator = 0
    for current_true, current_predicted in zip(y_true, y_predicted):
        round_predicted = int(np.round(current_predicted))
        if current_true == round_predicted:
            numerator += 1
        denominator += 1
    return numerator/denominator

def open_gridsearch_results_ML(input_file = ML_GRIDSEARCH_PARAMETERS):

    """
    Open and process the ML gridsearch results
    """
    proper_dictionary = {}
    opened_file = open(input_file, "r").readlines()[1:]
    for row in opened_file:
        row = row.replace(PARAGRAPH_SEP, "").split(CSV_SEP)
        if row[0] not in list(proper_dictionary):
            proper_dictionary[row[0]] = {}
        if row[1] not in list(proper_dictionary[row[0]]):
            proper_dictionary[row[0]][row[1]] = {}
        if row[2] not in list(proper_dictionary[row[0]][row[1]]):
            proper_dictionary[row[0]][row[1]][row[2]] = row[3:]
        else:
            proper_dictionary[row[0]][row[1]][row[2]] = row[3:]
    return proper_dictionary

def object_converter(input_object):

    """
    Check if and object can be converted to int, float or None
    """
    import ast
    if input_object == "0.0":
        return 0
    if input_object == "None":
        return None
    try:
        output_object = int(input_object)
        return output_object
    except:
        try:
            output_object = float(input_object)
            return output_object
        except: return str(input_object)

def correct_eli5_header(input_table, input_original_header):

    """
    Edit eli5 explain weights output to yield proper feature names and split weights and standard deviation colums
    """
    processed_table = []
    for row in input_table.split(PARAGRAPH_SEP)[10:]:
        split_row = row.split()
        processed_table.append([split_row[3], split_row[0], split_row[2]])
    
    processed_dataframe = pd.DataFrame(processed_table, columns = ["feature","weight","std"])

    replace_dictionary = {}
    for index, entry in enumerate(input_original_header):
        replace_dictionary["x" + str(index)] = entry
    return processed_dataframe.replace({"feature": replace_dictionary})

def evaluate_ML_feature_importance(preprocessing, input_data_dictionary, input_parameters_dictionary, \
                        classification_dictionary, regression_dictionary, \
                        verbose = True, write_mode = True):

    """
    Deploy the ML pipeline after gridsearch
    """
    data_dictionary = input_data_dictionary[preprocessing]
    parameters_dictionary = input_parameters_dictionary[preprocessing]
    for current_target in POSSIBLE_TARGETS:
        if current_target == "full_agreement":
            problem_type = "classification"
            current_target_name = "Full-agreement"
            metrics = METRICS_CLASSIFICATION
            ML_dictionary = classification_dictionary

        elif current_target != "full_agreement":
            continue
            current_target_name = current_target
            problem_type = "regression"
            metrics = METRICS_REGRESSION
            ML_dictionary = regression_dictionary

        for current_method in ML_dictionary.keys():
            searched_parameters = {x.split(":")[0]: object_converter(x.split(":")[1]) for x in parameters_dictionary[current_method][current_target_name]}
            if verbose == True:
                print("Currently evaluating importance for dataset", preprocessing, "with method", current_method, "for the target", current_target_name)
            base_predictor = ML_dictionary[current_method][0]
            parameters = {**ML_dictionary[current_method][1], **searched_parameters}
            predictor = base_predictor(**parameters)
            current_data_dictionary = prepare_dataset(data_dictionary[0], data_dictionary[1], sample_mode = False, \
                                                        target_column = current_target, task_type = problem_type)
            current_header = list(current_data_dictionary["train_features"])
            fitted_predictor = predictor.fit(current_data_dictionary["train_features"], current_data_dictionary["train_class"].values.ravel())
            permutator = PermutationImportance(fitted_predictor, cv = "prefit", random_state = RANDOM_STATE).fit(current_data_dictionary["test_features"], current_data_dictionary["test_class"].values.ravel())
            weights_table = eli5.format_as_text(eli5.explain_weights(permutator, top = current_data_dictionary["test_features"].shape[1]))
            weights_dataframe = correct_eli5_header(weights_table, current_header)
            output_name = FEATURE_IMPORTANCE_FOLDER + SYSTEM_SEP + FEATURE_IMPORTANCE_START + \
                    INTERMEDIATE_SEP + current_method + INTERMEDIATE_SEP + current_target_name + \
                    CSV_TERMINATION
            if write_mode == True:
                weights_dataframe.to_csv(output_name, sep = CSV_SEP, index = False)

ML_dictionary_classification = {"RF": [RandomForestClassifier, {"random_state": RANDOM_STATE, "n_jobs": -1, "max_features": "auto", "bootstrap": True}],
                "ETC": [ExtraTreesClassifier, {"random_state": RANDOM_STATE, "n_jobs": -1}], 
                "SVM": [LinearSVC, {"random_state": RANDOM_STATE}],
                "SGD": [SGDClassifier, {"random_state": RANDOM_STATE, "n_jobs": -1}],\
                "KNN": [KNeighborsClassifier, {"n_jobs": -1}], \
                "XGB": [xgb.XGBClassifier, {"n_jobs": -1, "random_state": RANDOM_STATE}]}

ML_dictionary_regression = {"RF": [RandomForestRegressor, {"random_state": RANDOM_STATE, "n_jobs": -1, "max_features": "auto", "bootstrap": True}],
                "ETC": [ExtraTreesRegressor, {"random_state": RANDOM_STATE, "n_jobs": -1}], 
                "SVM": [LinearSVR, {"random_state": RANDOM_STATE}],
                "SGD": [SGDRegressor, {"random_state": RANDOM_STATE}],\
                "KNN": [KNeighborsRegressor, {"n_jobs": -1}], \
                "XGB": [xgb.XGBRegressor, {"n_jobs": -1, "random_state": RANDOM_STATE}]}


preprocessing_mode = "PCA_fillna"

ML_gridsearch_results = open_gridsearch_results_ML()
evaluate_ML_feature_importance(preprocessing_mode, DATASETS_DICTIONARY_NO_CONCENTRATION, \
                    ML_gridsearch_results, ML_dictionary_classification, \
                    ML_dictionary_regression, write_mode = True)


"""
DL related code
"""

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


def DL_prepare_dataset(file = '', method = 'full_agreement', \
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
                    problem_type = "classification"):

    """
    Train the dataset
    """
    names, features, target = DL_prepare_dataset(file = dataset, method = method, \
                                                sample_fraction_mode = False, \
                                                task_type = problem_type)
    current_header = list(features)
    fitted_predictor = input_model.model.fit(x = features, y = target, epochs = 125, validation_split = 0.10)
    return fitted_predictor, current_header

def locate_best_parameters_DL(input_file = DL_GRIDSEARCH_PARAMETERS, target_col = "Target", \
                            usable_dataset = "PCA_fillna"):

    """
    Yield the the unique best architectures for each configuration
    """
    import pandas as pd
    opened_file = pd.read_csv(input_file, sep = CSV_SEP, header = 0)
    unique_targets = list(opened_file[target_col].unique())
    output_dictionary = {}
    for current_target in unique_targets:
        current_subset = opened_file.loc[opened_file[target_col] == current_target]
        best_architectures = [[int(y) for y in x.split("-")] for x in list(current_subset["Architecture"].unique())]
        dropout_rate = current_subset["Dropout Rate"].value_counts().idxmax()
        output_dictionary[current_target] = {"architecture": best_architectures, \
                                            "dropout_rate": [dropout_rate], \
                                            "dataset": [usable_dataset], \
                                            "target": [current_target]}
    return output_dictionary

DL_best_parameters = locate_best_parameters_DL()

for target_column in POSSIBLE_TARGETS:
    if target_column == "full_agreement":
        problem_type = "classification"
    else:
        problem_type = "regression"
        continue

    current_best_parameters = DL_best_parameters[target_column]
    train = DATASET_FOLDER + SYSTEM_SEP + preprocessing_mode.split(INTERMEDIATE_SEP)[0] + INTERMEDIATE_SEP + \
                "train" + INTERMEDIATE_SEP + preprocessing_mode.split(INTERMEDIATE_SEP)[1] + ".csv"
    test = DATASET_FOLDER + SYSTEM_SEP + preprocessing_mode.split(INTERMEDIATE_SEP)[0] + INTERMEDIATE_SEP + \
                "test" + INTERMEDIATE_SEP + preprocessing_mode.split(INTERMEDIATE_SEP)[1] + ".csv"
    for current_architecture in current_best_parameters["architecture"]:
        
        input_features = 1347
        raw_model =  neural_network(current_architecture, input_features, \
                                    dropout_rate = current_best_parameters["dropout_rate"][0], prediction_mode = problem_type)
        optimizer = tf.keras.optimizers.Adam(0.0001)
        output_name = FEATURE_IMPORTANCE_FOLDER + SYSTEM_SEP + FEATURE_IMPORTANCE_START + INTERMEDIATE_SEP + \
                        "keras" + INTERMEDIATE_SEP + INTERMEDIATE_SEP.join([str(x) for x in current_architecture]) + \
                        INTERMEDIATE_SEP + str(current_best_parameters["dropout_rate"][0]) + INTERMEDIATE_SEP + target_column + CSV_TERMINATION
        if problem_type == "classification":
            raw_model.model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
            classification_predictor, usable_header = model_training(train, \
                                        target_column, raw_model, \
                                        problem_type = problem_type)
            names, test_features, test_target = DL_prepare_dataset(file = test, method = target_column, task_type = "classification", sample_fraction_mode = False)
            permutator = PermutationImportance(classification_predictor.model, cv = "prefit", scoring = make_scorer(tailored_accuracy), \
                                    random_state = RANDOM_STATE).fit(test_features, test_target.values.ravel())
            weights_table = eli5.format_as_text(eli5.explain_weights(permutator, top = input_features))
            weights_dataframe = correct_eli5_header(weights_table, usable_header)
            weights_dataframe.to_csv(output_name, sep = CSV_SEP, index = False)

        elif problem_type == "regression":
            raw_model.model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse','mae'])
            regression_predictor, usable_header = model_training(train, \
                                                    target_column, raw_model, \
                                                    problem_type = problem_type)

            names, test_features, test_target = DL_prepare_dataset(file = test, method = target_column, task_type = "regression", sample_fraction_mode = False)
            permutator = PermutationImportance(regression_predictor.model, cv = "prefit", scoring = make_scorer(mean_squared_error), \
                                    random_state = RANDOM_STATE).fit(test_features, test_target.values.ravel())
            weights_table = eli5.format_as_text(eli5.explain_weights(permutator, top = input_features))
            weights_dataframe = correct_eli5_header(weights_table, usable_header)
            weights_dataframe.to_csv(output_name, sep = CSV_SEP, index = False)