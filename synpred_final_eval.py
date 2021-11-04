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
                            SUPPORT_FOLDER, POSSIBLE_TARGETS, DL_ENSEMBLE_PARAMETERS, \
                            EVALUATION_DL_FOLDER, PARAGRAPH_SEP, DATASETS_DICTIONARY, \
                            DEFAULT_LOCATION
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

def replace_outliers(input_row, multiply_number = 10):

    """
    From an input table, replace the outlier values with the average, if they exceed the average by too much
    """
    input_list = list(input_row)
    output_row = []
    for index, entry in enumerate(input_list):
        subset_row = input_list[0:index] + input_list[index + 1:]
        calculate_mean = np.mean(subset_row)
        if len(str(entry).split(".")[0]) >= 3:
            output_row.append(0.0)
        elif abs(entry) > abs(calculate_mean * multiply_number):
            output_row.append(calculate_mean)
        else:
            output_row.append(entry)
    return output_row

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
                                                    sample_mode = False, target_column = current_target, task_type = self.problem_type, \
                                                    final_mode = True, final_reduction = "PCA", final_preprocessing = "fillna")
        self.classes_dictionary = {"train_class": self.data_dictionary["train_class"], \
                                    "test_class": self.data_dictionary["test_class"], \
                                    "cell_class": self.data_dictionary["cell_class"], \
                                    "drugs_class": self.data_dictionary["drugs_class"], \
                                    "combo_class": self.data_dictionary["combo_class"]}
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
        output_name = DEFAULT_LOCATION + SYSTEM_SEP + "individual_evaluation" + SYSTEM_SEP + self.target + "_individual_metrics.csv"
        if self.problem_type == "classification":
            predictors_list = list(self.predictions_dictionary_classes)
            output_table = []
            for entry in predictors_list:
                split_entry = entry.split("_")
                current_subset = split_entry[-2]
                if split_entry[0] == "DL":
                    current_model = split_entry[0] + "_" + split_entry[1]
                else:
                    current_model = split_entry[3]
                current_class_subset = self.classes_dictionary[current_subset + "_class"]
                agreement_prediction = [int(np.round(x)) for x in self.predictions_dictionary_classes[entry]]
                current_results = model_evaluation(current_class_subset, agreement_prediction, \
                        subset_type = current_subset, verbose = False, \
                        write_mode = False, task_type = self.problem_type)
                output_table.append([entry] + current_results)
            current_dataframe = pd.DataFrame(output_table, columns = ["ID","Accuracy", "Precision", "Recall", "AUC", "F1-value"])
            current_dataframe.to_csv(output_name, index = False)

        elif self.problem_type == "regression":
            predictors_list = list(self.predictions_dictionary_probs)
            output_table = []
            for entry in predictors_list:
                split_entry = entry.split("_")
                current_subset = split_entry[-2]
                if split_entry[0] == "DL":
                    current_model = split_entry[0] + "_" + split_entry[1]
                else:
                    current_model = split_entry[3]
                current_target_subset = self.classes_dictionary[current_subset + "_class"]
                target_prediction_column = [float(x) for x in self.predictions_dictionary_probs[entry]]
                current_results = model_evaluation(current_target_subset, target_prediction_column, \
                        subset_type = current_subset, verbose = False, \
                        write_mode = False, task_type = self.problem_type)
                output_table.append([entry] + current_results)
            current_dataframe = pd.DataFrame(output_table, columns = ["ID","RMSE", "MSE", "Pearson", "r2", "MAE", "Spearman"])
            current_dataframe.to_csv(output_name, index = False)

for target in POSSIBLE_TARGETS:
    if target == "full_agreement":
        prob_type = "classification"
    else:
        prob_type = "regression"
    regular_object = ensemble(REDEPLOYMENT_FOLDER, DL_SAVED_MODELS, "non_drop", \
                                    problem_type = prob_type, write_class_pickle = False, current_target = target)
    regular_object.load_pred_dicts()