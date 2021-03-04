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
                            INTERMEDIATE_SEP, REDEPLOYMENT_FOLDER
import xgboost as xgb
import pickle
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

def prepare_dataset(input_train, input_test, drop_columns = DROPPABLE_COLUMNS, \
                    target_column = TARGET_CLASS_COLUMN, subset_size = 0):
    
    if subset_size != 0:
        train_dataset = pd.read_csv(input_train, sep = CSV_SEP, header = 0, nrows = subset_size)
        test_dataset = pd.read_csv(input_test, sep = CSV_SEP, header = 0, nrows = subset_size)
    else:
        train_dataset = pd.read_csv(input_train, sep = CSV_SEP, header = 0)
        test_dataset = pd.read_csv(input_test, sep = CSV_SEP, header = 0)
    train_class = train_dataset[target_column]
    train_features = train_dataset.drop(drop_columns, axis = 1)

    
    test_class = test_dataset[target_column]
    test_features = test_dataset.drop(drop_columns, axis = 1)

    return {"train_features": train_features, "train_class": train_class, \
            "test_features": test_features, "test_class": test_class}

def deploy_ML_pipeline(input_data_dictionary, input_ML_dictionary, verbose = True, save_model = True):

    for current_dataset in input_data_dictionary.keys():
        for current_method in input_ML_dictionary.keys():
            if verbose == True:
                print("Currently evaluating dataset", current_dataset, "with method",current_method)
            classifier = input_ML_dictionary[current_method]
            data_dictionary = prepare_dataset(input_data_dictionary[current_dataset][0], input_data_dictionary[current_dataset][1])
            classifier.fit(data_dictionary["train_features"], data_dictionary["train_class"].values.ravel())
            predicted_train = classifier.predict(data_dictionary["train_features"])
            predicted_test = classifier.predict(data_dictionary["test_features"])
            output_name = current_dataset + INTERMEDIATE_SEP + current_method + INTERMEDIATE_SEP
            model_evaluation(data_dictionary["train_class"], predicted_train, \
                                verbose = True, write_mode = True, subset_type = output_name + "train")

            model_evaluation(data_dictionary["test_class"], predicted_test, \
                                verbose = True, write_mode = True, subset_type = output_name + "test")
            model_name = REDEPLOYMENT_FOLDER + SYSTEM_SEP + output_name + ".pkl"
            with open(model_name,'wb') as output_pkl:
                pickle.dump(classifier, output_pkl)


datasets_dictionary = {"PCA": [TRAIN_DATASET_PCA, TEST_DATASET_PCA]}#, \
                        #"PCA_drop": [TRAIN_DATASET_PCA_DROP, TEST_DATASET_PCA_DROP]}#, \
                        #"autoencoder": [TRAIN_DATASET_AUTOENCODER, TEST_DATASET_AUTOENCODER], \
                        #"autoencoder_drop": [TRAIN_DATASET_AUTOENCODER_DROP, TEST_DATASET_AUTOENCODER_DROP], \
                        #"PCA_custom_drugs": [TRAIN_DATASET_PCA_CUSTOM_DRUGS, TEST_DATASET_PCA_CUSTOM_DRUGS], \
                        #"PCA_drop_custom_drugs": [TRAIN_DATASET_PCA_DROP_CUSTOM_DRUGS, TEST_DATASET_PCA_DROP_CUSTOM_DRUGS], \
                        #"autoencoder_custom_drugs": [TRAIN_DATASET_AUTOENCODER_CUSTOM_DRUGS, TEST_DATASET_AUTOENCODER_CUSTOM_DRUGS], \
                        #"autoencoder_drop_custom_drugs": [TRAIN_DATASET_AUTOENCODER_DROP_CUSTOM_DRUGS, TEST_DATASET_AUTOENCODER_DROP_CUSTOM_DRUGS]
                        #}

ML_dictionary = {"MLP": MLPClassifier(random_state = RANDOM_STATE, verbose = True),
                    "RF": RandomForestClassifier(random_state = RANDOM_STATE, n_jobs = -1, max_depth = None, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 1000),
                    "ETC": ExtraTreesClassifier(random_state = RANDOM_STATE, n_jobs = -1, max_depth = None, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 1000), 
                    "SVM": LinearSVC(random_state = RANDOM_STATE, C = 0.5),
                    "SGD": SGDClassifier(random_state = RANDOM_STATE, n_jobs = -1, alpha = 0.00001, penalty = "l1"),
                    "KNN": KNeighborsClassifier(n_jobs = -1, n_neighbors = 25), \
                    "XGB": xgb.XGBClassifier(n_jobs = -1, random_state = RANDOM_STATE, alpha = 0.0, max_depth = 6, n_estimators = 1000)
                    }

deploy_ML_pipeline(datasets_dictionary, ML_dictionary)