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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, roc_auc_score
from DEC_variables import SYSTEM_SEP, PARAGRAPH_SEP, CSV_SEP, \
                            TRAIN_DATASET_PCA, TEST_DATASET_PCA, \
                            TRAIN_DATASET_CUSTOM_DRUG, TEST_DATASET_CUSTOM_DRUG, \
                            TRAIN_DATASET_PCA_DROP, TEST_DATASET_PCA_DROP, \
                            TRAIN_DATASET_AUTOENCODER, TEST_DATASET_AUTOENCODER, \
                            TRAIN_DATASET_AUTOENCODER_DROP, TEST_DATASET_AUTOENCODER_DROP, \
                            TRAIN_DATASET_PCA_CUSTOM_DRUGS, TEST_DATASET_PCA_CUSTOM_DRUGS, \
                            TRAIN_DATASET_PCA_DROP_CUSTOM_DRUGS, TEST_DATASET_PCA_DROP_CUSTOM_DRUGS, \
                            TRAIN_DATASET_AUTOENCODER_CUSTOM_DRUGS, TEST_DATASET_AUTOENCODER_CUSTOM_DRUGS, \
                            TRAIN_DATASET_AUTOENCODER_DROP_CUSTOM_DRUGS, TEST_DATASET_AUTOENCODER_DROP_CUSTOM_DRUGS, \
                            DROPPABLE_COLUMNS, TARGET_CLASS_COLUMN, RANDOM_STATE, EVALUATION_NON_DL_FOLDER, \
                            INTERMEDIATE_SEP, REDEPLOYMENT_FOLDER, SUPPORT_FOLDER, RANDOM_STATE
import xgboost as xgb
import pickle
from sklearn.model_selection import GridSearchCV
from DEC_support_functions import prepare_dataset
import random
__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
def model_evaluation(input_class, input_predictions, \
                        subset_type = "test", verbose = False, \
                        write_mode = False):
    
    output_file_name = EVALUATION_NON_DL_FOLDER + SYSTEM_SEP + subset_type + ".csv"
    accuracy = accuracy_score(input_class, input_predictions)
    precision = precision_score(input_class, input_predictions)
    recall = recall_score(input_class, input_predictions)
    auc = roc_auc_score(input_class, input_predictions)
    f1_value = f1_score(input_class, input_predictions)
    if verbose == True:
        print("Accuracy:", round(accuracy, 2), "\n",
               "Precision:" , round(precision, 2), "\n",
            "Recall:", round(recall, 2), "\n",
            "AUC:", round(auc, 2), "\n",
            "F1-score:", round(f1_value, 2), "\n")
    if write_mode == True:
        with open(output_file_name, "w") as output_file:
            output_file.write("Metric,Value\n")
            output_file.write("Accuracy," + str(accuracy) + "\n")
            output_file.write("Precision," + str(precision) + "\n")
            output_file.write("Recall," + str(recall) + "\n")
            output_file.write("AUC," + str(auc) + "\n")
            output_file.write("F1-score," + str(f1_value) + "\n")
    return [accuracy, precision, recall, auc, f1_value]

def prepare_dataset(input_train, input_test, drop_columns = DROPPABLE_COLUMNS, \
                    target_column = TARGET_CLASS_COLUMN, subset_size = 0, sample_size = 1.0):
    
    if subset_size != 0:
        train_dataset = pd.read_csv(input_train, sep = CSV_SEP, header = 0, nrows = subset_size)
        test_dataset = pd.read_csv(input_test, sep = CSV_SEP, header = 0, nrows = subset_size)
    else:
        train_dataset = pd.read_csv(input_train, sep = CSV_SEP, header = 0).sample(frac = sample_size, axis = 0)
        test_dataset = pd.read_csv(input_test, sep = CSV_SEP, header = 0)
    train_class = train_dataset[target_column]
    train_features = train_dataset.drop(drop_columns, axis = 1)

    
    test_class = test_dataset[target_column]
    test_features = test_dataset.drop(drop_columns, axis = 1)

    return {"train_features": train_features, "train_class": train_class, \
            "test_features": test_features, "test_class": test_class}

def deploy_ML_pipeline(input_data_dictionary, input_ML_dictionary, verbose = True, save_model = True, output_file_name = SUPPORT_FOLDER + SYSTEM_SEP + "ML_gridsearch_FEV.csv"):

    metrics = ["Accuracy","Precision","Recall","AUC","F1-value"]
    with open(output_file_name, "w") as output_file:
        for current_dataset in input_data_dictionary.keys():
            for current_method in input_ML_dictionary.keys():
                if verbose == True:
                    print("Currently evaluating dataset", current_dataset, "with method",current_method)
                classifier = input_ML_dictionary[current_method]
                current_gridsearch = GridSearchCV(classifier[0], param_grid = classifier[1], scoring = 'accuracy', 
                            cv = 3, 
                            verbose = 2,
                            n_jobs = -1)
                data_dictionary = prepare_dataset(input_data_dictionary[current_dataset][0], input_data_dictionary[current_dataset][1], sample_size = 0.10)
                current_gridsearch.fit(data_dictionary["train_features"], data_dictionary["train_class"].values.ravel())
                classifier_object = current_gridsearch.best_estimator_
                output_params_dictionary = current_gridsearch.best_params_
                current_row = current_dataset + CSV_SEP + current_method + CSV_SEP
                for current_key in output_params_dictionary.keys():
                    current_row += str(current_key) + ":" + str(output_params_dictionary[current_key]) + CSV_SEP

                output_file.write(current_row[0:-1] + PARAGRAPH_SEP)
                predicted_train = classifier_object.predict(data_dictionary["train_features"])
                predicted_test = classifier_object.predict(data_dictionary["test_features"])
                output_name = current_dataset + INTERMEDIATE_SEP + current_method + INTERMEDIATE_SEP
                model_train_results = model_evaluation(data_dictionary["train_class"], predicted_train, \
                                    verbose = False, write_mode = False, subset_type = output_name + "train")

                model_test_results = model_evaluation(data_dictionary["test_class"], predicted_test, \
                                    verbose = False, write_mode = False, subset_type = output_name + "test")
                for current_metric, train_metric, test_metric in zip(metrics, model_train_results, model_test_results):
                    output_file.write(str(current_metric) + CSV_SEP + "Train:" + str(np.round(train_metric,4)) + CSV_SEP +  "Test:" + str(np.round(test_metric,4)) + PARAGRAPH_SEP)


datasets_dictionary = {"PCA": [TRAIN_DATASET_PCA, TEST_DATASET_PCA], \
                        "PCA_drop": [TRAIN_DATASET_PCA_DROP, TEST_DATASET_PCA_DROP], \
                        "autoencoder": [TRAIN_DATASET_AUTOENCODER, TEST_DATASET_AUTOENCODER], \
                        "autoencoder_drop": [TRAIN_DATASET_AUTOENCODER_DROP, TEST_DATASET_AUTOENCODER_DROP]}#, \
                        #"PCA_custom_drugs": [TRAIN_DATASET_PCA_CUSTOM_DRUGS, TEST_DATASET_PCA_CUSTOM_DRUGS], \
                        #"PCA_drop_custom_drugs": [TRAIN_DATASET_PCA_DROP_CUSTOM_DRUGS, TEST_DATASET_PCA_DROP_CUSTOM_DRUGS], \
                        #"autoencoder_custom_drugs": [TRAIN_DATASET_AUTOENCODER_CUSTOM_DRUGS, TEST_DATASET_AUTOENCODER_CUSTOM_DRUGS], \
                        #"autoencoder_drop_custom_drugs": [TRAIN_DATASET_AUTOENCODER_DROP_CUSTOM_DRUGS, TEST_DATASET_AUTOENCODER_DROP_CUSTOM_DRUGS]
                        #}

ML_dictionary = {"RF": [RandomForestClassifier(random_state = RANDOM_STATE, n_jobs = -1, max_features = "auto", bootstrap = True), \
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

deploy_ML_pipeline(datasets_dictionary, ML_dictionary, save_model = False)