"""
Script to deploy ML methods with gridsearch
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, roc_auc_score
from synpred_variables import SYSTEM_SEP, PARAGRAPH_SEP, CSV_SEP, \
                            DATASETS_DICTIONARY, DROPPABLE_COLUMNS, TARGET_CLASS_COLUMN, \
                            RANDOM_STATE, EVALUATION_NON_DL_FOLDER, \
                            INTERMEDIATE_SEP, REDEPLOYMENT_FOLDER, SUPPORT_FOLDER, RANDOM_STATE, \
                            DATASETS_DICTIONARY_NO_CONCENTRATION, \
                            METRICS_CLASSIFICATION, METRICS_REGRESSION, POSSIBLE_TARGETS
import xgboost as xgb
import pickle
from sklearn.model_selection import GridSearchCV
from synpred_support_functions import prepare_dataset, model_evaluation
import random
__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def deploy_ML_pipeline(input_data_dictionary, input_ML_dictionary, verbose = True, \
                        save_model = True, \
                        output_file_name = SUPPORT_FOLDER + SYSTEM_SEP + "ML_gridsearch.csv"):

    """
    Deploy ML pipeline, with gridsearch, on the dataset
    """
    with open(output_file_name, "w") as output_file:
        for current_dataset in input_data_dictionary.keys():
            for current_target in POSSIBLE_TARGETS:
                if current_target == "full_agreement":
                    problem_type = "classification"
                    metrics = METRICS_CLASSIFICATION
                elif current_target != "full_agreement":
                    problem_type = "regression"
                    metrics = METRICS_REGRESSION
                methods_dictionary = input_ML_dictionary[problem_type]
                for current_method in methods_dictionary.keys():
                    
                    if verbose == True:
                        print("Currently evaluating dataset", current_dataset, "with method", current_method)
                    classifier = methods_dictionary[current_method]
                    if problem_type == "classification":
                        current_gridsearch = GridSearchCV(classifier[0], param_grid = classifier[1], scoring = 'accuracy', 
                                    cv = 3, 
                                    verbose = 2,
                                    n_jobs = -1)
                    elif problem_type == "regression":
                        current_gridsearch = GridSearchCV(classifier[0], param_grid = classifier[1], scoring = 'neg_root_mean_squared_error', 
                                    cv = 3, 
                                    verbose = 2,
                                    n_jobs = -1)
                    data_dictionary = prepare_dataset(input_data_dictionary[current_dataset][0], \
                                                        input_data_dictionary[current_dataset][1], \
                                                        sample_size = 0.10, sample_mode = True, \
                                                        target_column = current_target, task_type = problem_type)
                    current_gridsearch.fit(data_dictionary["train_features"], data_dictionary["train_class"].values.ravel())
                    predictor_object = current_gridsearch.best_estimator_
                    output_params_dictionary = current_gridsearch.best_params_
                    current_row = current_dataset + CSV_SEP + current_method + CSV_SEP
                    for current_key in output_params_dictionary.keys():
                        current_row += str(current_key) + ":" + str(output_params_dictionary[current_key]) + CSV_SEP

                    output_file.write(current_row[0:-1] + PARAGRAPH_SEP)
                    predicted_train = predictor_object.predict(data_dictionary["train_features"])
                    predicted_test = predictor_object.predict(data_dictionary["test_features"])
                    output_name = current_dataset + INTERMEDIATE_SEP + current_method + INTERMEDIATE_SEP + current_target + INTERMEDIATE_SEP
                    model_train_results = model_evaluation(data_dictionary["train_class"], predicted_train, \
                                        verbose = False, write_mode = True, subset_type = output_name + "train", \
                                        task_type = problem_type)

                    model_test_results = model_evaluation(data_dictionary["test_class"], predicted_test, \
                                        verbose = False, write_mode = True, subset_type = output_name + "test", \
                                        task_type = problem_type)
                    for current_metric, train_metric, test_metric in zip(metrics, model_train_results, model_test_results):
                        output_file.write(str(current_metric) + CSV_SEP + "Train:" + \
                                    str(np.round(train_metric,4)) + CSV_SEP +  "Test:" + \
                                    str(np.round(test_metric,4)) + CSV_SEP + \
                                    problem_type + PARAGRAPH_SEP)

ML_dictionary_class = {"RF": [RandomForestClassifier(random_state = RANDOM_STATE, n_jobs = -1, max_features = "auto", bootstrap = True), \
                        {"n_estimators":[10,100,1000], "max_depth": [None, 1, 5], "min_samples_split": [2,5,10], \
                        "min_samples_leaf": [1, 2, 4]}],
                "ETC": [ExtraTreesClassifier(random_state = RANDOM_STATE, n_jobs = -1), \
                    {"n_estimators":[10,100,1000], "min_samples_split": [2,5,10], \
                    "max_depth": [None, 1, 5], "min_samples_leaf": [1, 2, 4]}], 
                "SVM": [LinearSVC(random_state = RANDOM_STATE), {"C":[0.1,0.5,1.0]}],
                "SGD": [SGDClassifier(random_state = RANDOM_STATE, n_jobs = -1), {"penalty": ["l2", "l1", "elasticnet"], \
                "alpha": [0.00001,0.0001,0.001]}],\
                "KNN": [KNeighborsClassifier(n_jobs = -1), {"n_neighbors": [2,5,10,25]}], \
                "XGB": [xgb.XGBClassifier(n_jobs = -1, random_state = RANDOM_STATE), {"max_depth":[2,6,10], \
                "alpha":[0,0.25,0.50], "n_estimators": [10,100,1000]}]}

ML_dictionary_regression = {"RF": [RandomForestRegressor(random_state = RANDOM_STATE, n_jobs = -1, max_features = "auto", bootstrap = True), \
                        {"n_estimators":[10,100,1000], "max_depth": [None, 1, 5], "min_samples_split": [2,5,10], \
                        "min_samples_leaf": [1, 2, 4]}],
                "ETC": [ExtraTreesRegressor(random_state = RANDOM_STATE, n_jobs = -1), \
                    {"n_estimators":[10,100,1000], "min_samples_split": [2,5,10], \
                    "max_depth": [None, 1, 5], "min_samples_leaf": [1, 2, 4]}], 
                "SVM": [LinearSVR(random_state = RANDOM_STATE), {"C":[0.1,0.5,1.0]}],
                "SGD": [SGDRegressor(random_state = RANDOM_STATE), {"penalty": ["l2", "l1", "elasticnet"], \
                "alpha": [0.00001,0.0001,0.001]}],\
                "KNN": [KNeighborsRegressor(n_jobs = -1), {"n_neighbors": [2,5,10,25]}], \
                "XGB": [xgb.XGBRegressor(n_jobs = -1, random_state = RANDOM_STATE), {"max_depth":[2,6,10], \
                "alpha":[0,0.25,0.50], "n_estimators": [10,100,1000]}]}

deploy_ML_pipeline(DATASETS_DICTIONARY_NO_CONCENTRATION, {"classification": ML_dictionary_class, "regression": ML_dictionary_regression}, save_model = False)