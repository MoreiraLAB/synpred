"""
Script to deploy tensorflow for Gridsearch
conda activate tf
tensorflow version 1.15
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
from DEC_variables import RANDOM_STATE, DROPPABLE_COLUMNS, TARGET_CLASS_COLUMN, \
                            CSV_SEP, TRAIN_DATASET_PCA, TEST_DATASET_PCA, \
                            FEATURES_IMPORTANCE_OUTPUT_FILE, PARAGRAPH_SEP
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import sys
import random
import numpy as np
import ast
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import eli5
from eli5.sklearn import PermutationImportance
import xgboost as xgb

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.compat.v1.set_random_seed(RANDOM_STATE)
c_type = 'all'

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


def neural_network(input_architecture = [2500]*2, input_features = 1347, \
                    dropout_rate = 0.0, activation_function = "relu"):

    classifier = Sequential()
    classifier.add(Dense(units=input_architecture[0],input_dim=input_features, \
                                activation = activation_function, \
                                kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                bias_regularizer = regularizers.l2(1e-4), \
                                activity_regularizer = regularizers.l2(1e-5)))
    for layer_size in input_architecture[1:]:
        classifier.add(Dense(units=layer_size, activation = activation_function, \
                                    kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                    bias_regularizer = regularizers.l2(1e-4), \
                                    activity_regularizer = regularizers.l2(1e-5)))
    classifier.add(Dense(units=1,activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(0.0001)
    classifier.compile(optimizer = optimizer,loss = 'binary_crossentropy',
                      metrics=['accuracy'])
    return classifier

def write_importance_file(input_assessment, model_type = "", output_name = FEATURES_IMPORTANCE_OUTPUT_FILE):

    output_name = FEATURES_IMPORTANCE_OUTPUT_FILE.split(".")[0] + "_" + model_type + ".csv"
    with open(output_name, "w") as output_file:
        features_dict = input_assessment["feature_importances"]["importances"]
        output_file.write("feature" + CSV_SEP + "weight" + \
            CSV_SEP + "std" + CSV_SEP + "value" + PARAGRAPH_SEP)
        for entry in features_dict:
            write_row = str(entry["feature"]) + CSV_SEP + str(entry["weight"]) + \
                CSV_SEP + str(entry["std"]) + CSV_SEP + str(entry["value"]) + PARAGRAPH_SEP
            output_file.write(write_row)

datasets_dictionary = {"PCA": [TRAIN_DATASET_PCA, TEST_DATASET_PCA]}

ML_dictionary = {"MLP": MLPClassifier(random_state = RANDOM_STATE, verbose = True),
                    "RF": RandomForestClassifier(random_state = RANDOM_STATE, n_jobs = -1, max_depth = None, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 1000),
                    "ETC": ExtraTreesClassifier(random_state = RANDOM_STATE, n_jobs = -1, max_depth = None, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 1000), 
                    "SVM": LinearSVC(random_state = RANDOM_STATE, C = 0.5),
                    "SGD": SGDClassifier(random_state = RANDOM_STATE, n_jobs = -1, alpha = 0.00001, penalty = "l1"),
                    "KNN": KNeighborsClassifier(n_jobs = -1, n_neighbors = 25), \
                    "XGB": xgb.XGBClassifier(n_jobs = -1, random_state = RANDOM_STATE, alpha = 0.0, max_depth = 6, n_estimators = 1000)
                    }

input_architecture = [2114,1057]
input_features = 1347
dropout_rate = 0.0
proper_dictionary = prepare_dataset(datasets_dictionary["PCA"][0],datasets_dictionary["PCA"][1])
#my_model = KerasClassifier(build_fn=neural_network)
for current_key in ML_dictionary.keys():
    print("Currently on:", current_key)
    my_model = ML_dictionary[current_key]
    my_model.fit(proper_dictionary["train_features"], proper_dictionary["train_class"].values.ravel()), \
    #                    epochs = 3, validation_split = 0.10)
    perm = PermutationImportance(my_model, random_state = RANDOM_STATE).fit(proper_dictionary["test_features"], \
                                        proper_dictionary["test_class"].values.ravel())
    feature_weights = eli5.format_as_dict(eli5.explain_weights(perm, feature_names = proper_dictionary["test_features"].columns.tolist(), top = input_features))
    write_importance_file(feature_weights, model_type = current_key)