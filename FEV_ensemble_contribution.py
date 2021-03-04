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
from tensorflow.keras.models import load_model 
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
import tensorflow as tf
import os
import random
import numpy as np
import ast
import pickle
import sklearn
from DEC_variables import REDEPLOYMENT_FOLDER, SYSTEM_SEP, \
                            CSV_SEP, INTERMEDIATE_SEP, \
                            DL_SAVED_MODELS, RANDOM_STATE, \
                            PKL_TERMINATION, TRAIN_DATASET_PCA, \
                            TEST_DATASET_PCA, TRAIN_DATASET_PCA_DROP, \
                            TEST_DATASET_PCA_DROP, SUPPORT_FOLDER
from DEC_support_functions import model_evaluation, prepare_dataset
import h5py

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.compat.v1.set_random_seed(RANDOM_STATE)
c_type = 'all'

class neural_network_class:

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


def locate_models(input_folder, folder_condition = "ML", secondary_condition = "PCA", third_condition = "drop"):

    """
    Output a list of the models available on folder according to condition
    """
    models_dictionary = {}
    for files in os.listdir(input_folder):

        if folder_condition == "ML":
            if files.endswith(PKL_TERMINATION):
                split_file = files.split(INTERMEDIATE_SEP)
                if split_file[0] == secondary_condition:
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
            int_mode = int(split_file[0])
            if int_mode % 2 == 0:
                temp_mode = "drop"
            elif int_mode % 2 != 0:
                temp_mode = "non_drop"
            if temp_mode == third_condition:
                model = load_model(os.path.join("./saved_model/",files))
                models_dictionary[folder_condition + INTERMEDIATE_SEP + files.split(".")[0]] = model

    return models_dictionary

class ensemble:

    def __init__(self, ML_models_folder, DL_models_folder, third_cond, write_class_pickle = True):

        self.mode = third_cond
        self.models_dict_ML = locate_models(ML_models_folder, folder_condition = "ML", \
                                            secondary_condition = "PCA", third_condition = third_cond)
        self.models_dict_DL = locate_models(DL_models_folder, folder_condition = "DL", \
                                            secondary_condition = "PCA", third_condition = third_cond)
        self.all_models_dict = dict(self.models_dict_ML, **self.models_dict_DL)
        self.preprocess = third_cond
        if third_cond == "drop":
            self.datasets_dictionary = {"PCA_drop": [TRAIN_DATASET_PCA_DROP, TEST_DATASET_PCA_DROP]}
            self.processing_type = "PCA_drop"
        elif third_cond == "non_drop":
            self.datasets_dictionary = {"PCA": [TRAIN_DATASET_PCA, TEST_DATASET_PCA]}
            self.processing_type = "PCA"

        self.data_dictionary = prepare_dataset(self.datasets_dictionary[self.processing_type][0], self.datasets_dictionary[self.processing_type][1])
        self.classes_dictionary = {"train_class": self.data_dictionary["train_class"], \
                                    "test_class": self.data_dictionary["test_class"]}
        self.predictions_classes_path = SUPPORT_FOLDER + SYSTEM_SEP + self.mode + \
                        INTERMEDIATE_SEP + "predictions_dictionary_classes.pkl"
        self.predictions_probs_path = SUPPORT_FOLDER + SYSTEM_SEP + self.mode + \
                        INTERMEDIATE_SEP + "predictions_dictionary_probs.pkl"

        self.class_pickle_path = SUPPORT_FOLDER + SYSTEM_SEP + self.mode + \
                        INTERMEDIATE_SEP + "classes_dictionary.pkl"
        if write_class_pickle == True:
            with open(self.class_pickle_path, 'wb') as class_pkl:
                pickle.dump(self.classes_dictionary, class_pkl, protocol=pickle.HIGHEST_PROTOCOL)

    def dict_load(self, file_name):

        return pickle.load(open(file_name, "rb"))

    def load_pred_dicts(self):

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
                elif mode == "ML":
                    try:
                        predict_function = classifier.predict_proba
                    except:
                        predict_funcation = classifier.predict

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
        #class_subset = self.classes_dictionary[subset + INTERMEDIATE_SEP + "class"]
        
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
                if current_table.shape[1] == 2:
                    current_table = current_table[:,1]
                try:
                    table_row = [x[0] for x in current_table.tolist()]
                except:
                    table_row = current_table.tolist()
                output_table.append(table_row)
        return np.array(output_table).transpose()

    def generate_tables(self, pred_mode = "probabilities"):

        self.train_predictions_table = self.join_methods_predictions(subset = "train", mode = pred_mode)
        self.test_predictions_table = self.join_methods_predictions(subset = "test", mode = pred_mode)


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

def neural_network(input_architecture = [50]*4, input_features = 11, \
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

input_architecture = [50]*4
input_features = 11
dropout_rate = 0.60

regular_object = ensemble(REDEPLOYMENT_FOLDER, DL_SAVED_MODELS, "non_drop", write_class_pickle = False)
regular_object.load_pred_dicts()
regular_object.generate_tables()

models_header = ["MLP","RF","KNN","SGD","XGB","ETC","SVM","DL_5","DL_1","DL_7","DL_3"]
proper_dictionary = {"train_features": pd.DataFrame(regular_object.train_predictions_table, columns = models_header), "train_class": regular_object.classes_dictionary["train_class"], \
            "test_features": pd.DataFrame(regular_object.test_predictions_table, columns = models_header), "test_class": regular_object.classes_dictionary["test_class"]}
print(proper_dictionary)
my_model = KerasClassifier(build_fn=neural_network)
my_model.fit(proper_dictionary["train_features"], proper_dictionary["train_class"].values.ravel(), \
                        epochs = 3, validation_split = 0.10)

perm = PermutationImportance(my_model, random_state = RANDOM_STATE).fit(proper_dictionary["test_features"], \
                                    proper_dictionary["test_class"].values.ravel())

feature_weights = eli5.format_as_dict(eli5.explain_weights(perm, feature_names = proper_dictionary["test_features"].columns.tolist(), top = input_features))
write_importance_file(feature_weights, model_type = "ensemble_importance")