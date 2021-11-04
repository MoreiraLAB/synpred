"""
Script to deploy ML methods
conda activate tf
tensorflow version 1.15
"""

import os
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
                                RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from synpred_variables import SYSTEM_SEP, PARAGRAPH_SEP, CSV_SEP, \
                            RANDOM_STATE, EVALUATION_NON_DL_FOLDER, \
                            INTERMEDIATE_SEP, REDEPLOYMENT_FOLDER, DATASETS_DICTIONARY_NO_CONCENTRATION, \
                            ML_GRIDSEARCH_PARAMETERS, METRICS_CLASSIFICATION, METRICS_REGRESSION, \
                            POSSIBLE_TARGETS
import xgboost as xgb
import pickle
from synpred_support_functions import prepare_dataset, model_evaluation
import random
__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def open_gridsearch_results(input_file = ML_GRIDSEARCH_PARAMETERS):

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

def deploy_ML_pipeline(preprocessing, input_data_dictionary, input_parameters_dictionary, \
                        classification_dictionary, regression_dictionary, \
                        verbose = True, save_model = True):
    
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
            current_target_name = current_target
            problem_type = "regression"
            metrics = METRICS_REGRESSION
            ML_dictionary = regression_dictionary

        for current_method in ML_dictionary.keys():
            print(parameters_dictionary)
            searched_parameters = {x.split(":")[0]: object_converter(x.split(":")[1]) for x in parameters_dictionary[current_method][current_target_name]}
            if verbose == True:
                print("Currently evaluating dataset", preprocessing, "with method", current_method, "for the target", current_target_name)
            base_predictor = ML_dictionary[current_method][0]
            parameters = {**ML_dictionary[current_method][1], **searched_parameters}
            predictor = base_predictor(**parameters)
            current_data_dictionary = prepare_dataset(data_dictionary[0], data_dictionary[1], sample_mode = False, \
                                                        target_column = current_target, task_type = problem_type)
            predictor.fit(current_data_dictionary["train_features"], current_data_dictionary["train_class"].values.ravel())
            predicted_train = predictor.predict(current_data_dictionary["train_features"])
            predicted_test = predictor.predict(current_data_dictionary["test_features"])
            output_name = preprocessing + INTERMEDIATE_SEP + current_method + INTERMEDIATE_SEP + current_target_name
            model_evaluation(current_data_dictionary["train_class"], predicted_train, \
                                verbose = True, write_mode = True, subset_type = output_name + INTERMEDIATE_SEP + "train" + INTERMEDIATE_SEP + current_target_name, \
                                task_type = problem_type)

            model_evaluation(current_data_dictionary["test_class"], predicted_test, \
                                verbose = True, write_mode = True, subset_type = output_name + INTERMEDIATE_SEP + "test" + INTERMEDIATE_SEP + current_target_name, \
                                task_type = problem_type)
            model_name = REDEPLOYMENT_FOLDER + SYSTEM_SEP + output_name + ".pkl"
            with open(model_name,'wb') as output_pkl:
                pickle.dump(predictor, output_pkl)

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

ML_gridsearch_results = open_gridsearch_results()
preprocessing_mode = "PCA_fillna"

deploy_ML_pipeline(preprocessing_mode, DATASETS_DICTIONARY_NO_CONCENTRATION, \
                    ML_gridsearch_results, ML_dictionary_classification, \
                    ML_dictionary_regression, save_model = False)