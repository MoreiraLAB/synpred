#!/usr/bin/env python

"""
Join the different models trained with an ensemble
"""

__author__ = "A.J. Preto"
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
from tensorflow.keras.models import load_model 
import pandas as pd
from sklearn import preprocessing
from datetime import datetime
import sys
import os
import random
import numpy as np
import ast
import pickle
import sklearn
from synpred_variables import REDEPLOYMENT_FOLDER, SYSTEM_SEP, \
                            CSV_SEP, INTERMEDIATE_SEP, \
                            DL_SAVED_MODELS, RANDOM_STATE, \
                            PKL_TERMINATION, DATASETS_DICTIONARY_NO_CONCENTRATION, \
                            SUPPORT_FOLDER, POSSIBLE_TARGETS
from synpred_support_functions import model_evaluation, prepare_dataset
import h5py

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.compat.v1.set_random_seed(RANDOM_STATE)

class neural_network_class:

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

def locate_models(input_folder, folder_condition = "ML", secondary_condition = "PCA", \
                    third_condition = "drop", target = ""):

    """
    Output a list of the models available on folder according to condition
    """
    if (target == "full_agreement") and (folder_condition == "ML"):
        target = "Full-agreement"
    elif (target == "full_agreement") and (folder_condition == "DL"):
        target = "agreement"
    models_dictionary = {}
    for files in os.listdir(input_folder):
        if folder_condition == "ML":
            if files.endswith(PKL_TERMINATION):
                split_file = files.split(INTERMEDIATE_SEP)
                if (split_file[0] == secondary_condition) and (split_file[-1].split(".")[0] == target):
                    model_name =  folder_condition + INTERMEDIATE_SEP + INTERMEDIATE_SEP.join(split_file[0:-1])
                    files_loc = input_folder + SYSTEM_SEP + files
                    if split_file[1] == third_condition:
                        pickle_file = open(files_loc, 'rb')
                        loaded_model = pickle.load(pickle_file)
                        models_dictionary[model_name] = loaded_model
                    elif split_file[1] != third_condition:
                        pickle_file = open(files_loc, 'rb')
                        loaded_model = pickle.load(pickle_file)
                        models_dictionary[model_name] = loaded_model

        if folder_condition == "DL":
            split_file = files.split(INTERMEDIATE_SEP)
            if split_file[-1].split(".")[0] == target:
                model = load_model(os.path.join("./saved_model/",files))
                models_dictionary[folder_condition + INTERMEDIATE_SEP + files.split(".")[0]] = model
    return models_dictionary

def replace_outliers(input_row, multiply_number = 10):

    """
    From an input table, replace the outlier values with the average, if they exceed the average by too much
    """
    input_list = list(input_row)
    output_row = []
    for index, entry in enumerate(input_list):
        subset_row = input_list[0:index] + input_list[index + 1:]
        calculate_mean = np.mean(subset_row)
        if abs(entry) > abs(calculate_mean * multiply_number):
            output_row.append(calculate_mean)
        else:
            output_row.append(entry)
    return output_row

class ensemble:

    def __init__(self, ML_models_folder, DL_models_folder, third_cond, write_class_pickle = True, \
                    current_target = "", problem_type = ""):

        self.problem_type = problem_type
        self.mode = third_cond
        self.target = current_target
        self.models_dict_ML = locate_models(ML_models_folder, folder_condition = "ML", \
                                            secondary_condition = "PCA", third_condition = third_cond, \
                                            target = current_target)
        self.models_dict_DL = locate_models(DL_models_folder, folder_condition = "DL", \
                                            secondary_condition = "PCA", third_condition = third_cond, \
                                            target = current_target)
        self.all_models_dict = dict(self.models_dict_ML, **self.models_dict_DL)
        self.preprocess = third_cond
        if third_cond == "drop":
            self.datasets_dictionary = {"PCA_drop": DATASETS_DICTIONARY_NO_CONCENTRATION["PCA_dropna"]}
            self.processing_type = "PCA_drop"
        elif third_cond == "non_drop":
            self.datasets_dictionary = {"PCA": DATASETS_DICTIONARY_NO_CONCENTRATION["PCA_fillna"]}
            self.processing_type = "PCA"
        self.data_dictionary = prepare_dataset(self.datasets_dictionary[self.processing_type][0], self.datasets_dictionary[self.processing_type][1], \
                                                    target_column = current_target, task_type = self.problem_type)
        self.classes_dictionary = {"train_class": self.data_dictionary["train_class"], \
                                    "test_class": self.data_dictionary["test_class"]}
        self.predictions_classes_path = SUPPORT_FOLDER + SYSTEM_SEP + self.mode + \
                        INTERMEDIATE_SEP + current_target + INTERMEDIATE_SEP + "predictions_dictionary_classes.pkl"
        self.predictions_probs_path = SUPPORT_FOLDER + SYSTEM_SEP + self.mode + \
                        INTERMEDIATE_SEP + current_target + INTERMEDIATE_SEP + "predictions_dictionary_probs.pkl"

        self.class_pickle_path = SUPPORT_FOLDER + SYSTEM_SEP + self.mode + \
                        INTERMEDIATE_SEP + current_target + INTERMEDIATE_SEP + "classes_dictionary.pkl"
        if write_class_pickle == True:
            with open(self.class_pickle_path, 'wb') as class_pkl:
                pickle.dump(self.classes_dictionary, class_pkl, \
                                protocol = pickle.HIGHEST_PROTOCOL)

    def dict_load(self, file_name):

        return pickle.load(open(file_name, "rb"))

    def load_pred_dicts(self):

        if self.problem_type == "classification":
            self.predictions_dictionary_classes = self.dict_load(self.predictions_classes_path)
        self.predictions_dictionary_probs = self.dict_load(self.predictions_probs_path)

    def target_generator(self, prediction_mode = "classes", write = True):

        if prediction_mode == "classes":
            self.predictions_dictionary_classes = {}
            for current_method in self.all_models_dict.keys():
                classifier = self.all_models_dict[current_method]
                mode = current_method.split(INTERMEDIATE_SEP)[0]
                if mode == "DL":
                    predict_function = classifier.predict_classes
                elif mode == "ML":
                    predict_function = classifier.predict

                self.predictions_dictionary_classes[current_method + INTERMEDIATE_SEP + "train_class"] = \
                        predict_function(self.data_dictionary["train_features"])
                self.predictions_dictionary_classes[current_method + INTERMEDIATE_SEP + "test_class"] = \
                        predict_function(self.data_dictionary["test_features"])

            if write == True:
                with open(self.predictions_classes_path, 'wb') as pred_pkl_classes:
                    pickle.dump(self.predictions_dictionary_classes, pred_pkl_classes, protocol=pickle.HIGHEST_PROTOCOL)

        if prediction_mode == "probabilities":
            self.predictions_dictionary_probs = {}
            for current_method in self.all_models_dict.keys():
                classifier = self.all_models_dict[current_method]
                mode = current_method.split(INTERMEDIATE_SEP)[0]
                if mode == "DL":
                    predict_function = classifier.predict
                elif (mode == "ML") and (self.problem_type == "classification"):
                    try:
                        predict_function = classifier.predict_proba
                    except:
                        predict_function = classifier.predict
                elif (mode == "ML") and (self.problem_type == "regression"):
                    predict_function = classifier.predict
                self.predictions_dictionary_probs[current_method + INTERMEDIATE_SEP + "train_class"] = \
                        predict_function(self.data_dictionary["train_features"])
                self.predictions_dictionary_probs[current_method + INTERMEDIATE_SEP + "test_class"] = \
                        predict_function(self.data_dictionary["test_features"])

            if write == True:
                with open(self.predictions_probs_path, 'wb') as pred_pkl_proba:
                    pickle.dump(self.predictions_dictionary_probs, pred_pkl_proba, protocol=pickle.HIGHEST_PROTOCOL)

    def join_methods_predictions(self, subset = "train", mode = "probabilities"):

        """
        Must be called after either target_generator or load_pred_dicts
        """
        
        if mode == "probabilities":
            usable_dictionary = self.predictions_dictionary_probs
        elif mode == "classes":
            usable_dictionary = self.predictions_dictionary_classes
        output_table = []
        for current_model in usable_dictionary.keys():
            current_method = current_model.split(INTERMEDIATE_SEP)[-2]
            preprocess_mod = current_model.split(INTERMEDIATE_SEP)[2]
            if preprocess_mod != "drop":
                preprocess_mod = "non_drop"
            if current_method == subset and preprocess_mod == self.preprocess:
                current_table = usable_dictionary[current_model]
                try:
                    if current_table.shape[1] == 2:
                        current_table = current_table[:,1]
                except:
                    current_table = current_table
                try:
                    table_row = [x[0] for x in current_table.tolist()]
                except:
                    table_row = current_table.tolist()
                output_table.append(table_row)
        return np.array(output_table).transpose()

    def generate_tables(self, pred_mode = "probabilities"):

        train_predictions_table = self.join_methods_predictions(subset = "train", mode = pred_mode)
        self.train_predictions_table = np.apply_along_axis(replace_outliers, axis = 1, arr = train_predictions_table)
        test_predictions_table = self.join_methods_predictions(subset = "test", mode = pred_mode)
        self.test_predictions_table = np.apply_along_axis(replace_outliers, axis = 1, arr = test_predictions_table)

    def mean_ensemble(self, input_array):
        return [np.round(x).astype(int) for x in list(input_array.mean(axis = 1))]

    def nn_ensemble(self, input_array):

        architectures = [[10]*2,[10]*3,[10]*4,[10]*5,[10]*7,\
                            [25]*2,[25]*3,[25]*4,[25]*5,[25]*7,\
                            [50]*2,[50]*3,[50]*4,[50]*5,[50]*7,\
                            [100]*2,[100]*3,[100]*4,[100]*5,[100]*7,\
                            [500]*2,[500]*3,[500]*4,[500]*5,[500]*7]
        dropout_rates = [0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]
        count = 1
        for current_architecture in architectures:
            for current_dropout_rate in dropout_rates:
                model_name = "ensemble" + INTERMEDIATE_SEP + str(count) + INTERMEDIATE_SEP + \
                                INTERMEDIATE_SEP.join([str(x) for x in current_architecture]) + \
                                INTERMEDIATE_SEP + "dropout" + INTERMEDIATE_SEP + str(current_dropout_rate) + \
                                INTERMEDIATE_SEP + self.mode
                nn_model =  neural_network_class(current_architecture, \
                                input_array.shape[1], dropout_rate = current_dropout_rate, \
                                prediction_mode = self.problem_type)
                count +=1
                optimizer = tf.keras.optimizers.Adam(0.0001)
                if self.problem_type == "classification":
                    nn_model.model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
                    nn_model.model.fit(x = input_array, \
                                y = self.classes_dictionary["train_class"], \
                                epochs = 5, validation_split = 0.10)
                    train_predictions = [int(np.round(x)) for x in nn_model.model.predict(self.train_predictions_table)]
                    test_predictions = [int(np.round(x)) for x in nn_model.model.predict(self.test_predictions_table)]
                elif self.problem_type == "regression":
                    nn_model.model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse','mae'])
                    nn_model.model.fit(x = self.train_predictions_table.astype(float), \
                                y = self.classes_dictionary["train_class"], \
                                epochs = 5, validation_split = 0.10)
                    train_predictions = [x for x in nn_model.model.predict(self.train_predictions_table.astype(float))]
                    test_predictions = [x for x in nn_model.model.predict(self.test_predictions_table.astype(float))]
                model_evaluation(train_predictions, self.classes_dictionary["train_class"], \
                            subset_type = model_name + INTERMEDIATE_SEP + str(5) + INTERMEDIATE_SEP + "epochs" + INTERMEDIATE_SEP + self.target + INTERMEDIATE_SEP + "train", verbose = True, \
                            write_mode = True, task_type = self.problem_type)
                model_evaluation(test_predictions, self.classes_dictionary["test_class"], \
                            subset_type = model_name + INTERMEDIATE_SEP + str(5) + INTERMEDIATE_SEP + "epochs" + INTERMEDIATE_SEP + self.target + INTERMEDIATE_SEP + "test", verbose = True, \
                            write_mode = True, task_type = self.problem_type)

    def ML_ensemble_models(self, input_array):

        if self.problem_type == "classification":
            from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
            from sklearn.linear_model import SGDClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            import xgboost as xgb
            ML_dictionary = {"RF": RandomForestClassifier(random_state = RANDOM_STATE),
                        "ETC": ExtraTreesClassifier(random_state = RANDOM_STATE), 
                        "SVM": SVC(random_state = RANDOM_STATE),
                        "SGD": SGDClassifier(random_state = RANDOM_STATE),
                        "KNN": KNeighborsClassifier(), \
                        "XGB": xgb.XGBClassifier(n_jobs = -1, random_state = RANDOM_STATE),
                        }
        elif self.problem_type == "regression":
            from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
            from sklearn.linear_model import SGDRegressor
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.svm import SVR
            import xgboost as xgb
            ML_dictionary = {"RF": RandomForestRegressor(random_state = RANDOM_STATE),
                        "ETC": ExtraTreesRegressor(random_state = RANDOM_STATE), 
                        "SVM": SVR(),
                        "SGD": SGDRegressor(random_state = RANDOM_STATE),
                        "KNN": KNeighborsRegressor(), \
                        "XGB": xgb.XGBRegressor(n_jobs = -1, random_state = RANDOM_STATE),
                        }
        for current_key in ML_dictionary.keys():

            current_model = ML_dictionary[current_key]
            current_model.fit(input_array, self.classes_dictionary["train_class"])
            model_evaluation(current_model.predict(self.train_predictions_table), self.classes_dictionary["train_class"], \
                            subset_type = "ensemble" + INTERMEDIATE_SEP + current_key + INTERMEDIATE_SEP + self.mode + INTERMEDIATE_SEP + self.target + INTERMEDIATE_SEP + "train", verbose = True, \
                            write_mode = True, task_type = self.problem_type)
            model_evaluation(current_model.predict(self.test_predictions_table), self.classes_dictionary["test_class"], \
                            subset_type = "ensemble" + INTERMEDIATE_SEP + current_key + INTERMEDIATE_SEP + self.mode + INTERMEDIATE_SEP + self.target + INTERMEDIATE_SEP + "test", verbose = True, \
                            write_mode = True, task_type = self.problem_type)

    def deploy_ensemble(self):

        model_evaluation(self.mean_ensemble(self.train_predictions_table), self.classes_dictionary["train_class"], \
                            subset_type = "ensemble_mean" + INTERMEDIATE_SEP + self.mode + INTERMEDIATE_SEP + self.target + INTERMEDIATE_SEP +  "train", verbose = True, \
                            write_mode = True, task_type = self.problem_type)
        model_evaluation(self.mean_ensemble(self.test_predictions_table), self.classes_dictionary["test_class"], \
                            subset_type = "ensemble_mean" + INTERMEDIATE_SEP + self.mode + INTERMEDIATE_SEP + self.target + INTERMEDIATE_SEP + "test", verbose = True, \
                            write_mode = True, task_type = self.problem_type)

        self.nn_ensemble(self.train_predictions_table)
        
        self.ML_ensemble_models(self.train_predictions_table)


for target in POSSIBLE_TARGETS:
    if target == "full_agreement":
        prob_type = "classification"
    else:
        prob_type = "regression"
    regular_object = ensemble(REDEPLOYMENT_FOLDER, DL_SAVED_MODELS, "non_drop", write_class_pickle = False, \
                                current_target = target, problem_type = prob_type)

    regular_object.target_generator(prediction_mode = "probabilities")
    regular_object.target_generator(prediction_mode = "classes")

    regular_object.load_pred_dicts()
    regular_object.generate_tables()
    regular_object.deploy_ensemble()