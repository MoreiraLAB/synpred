#!/usr/bin/env python

"""
Deploy the previously loaded model and write the results onto a .csv file
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SYNPRED"

import pickle
import pandas as pd
from numpy.random import seed
import random
import sys
import pickle
import os
from standalone_variables import HOME, SUPPORT_FOLDER, SYSTEM_SEP, SEP, \
                        CELL_LINES_COLUMN, PROCESSED_TERMINATION, \
                        TENSORFLOW_MODEL_PATH_1, TENSORFLOW_MODEL_PATH_2, \
                        TENSORFLOW_MODEL_PATH_3, TENSORFLOW_MODEL_PATH_4, \
                        TENSORFLOW_MODEL_PATH_ENSEMBLE, XGB_MODEL_PATH, \
                        SVM_MODEL_PATH, KNN_MODEL_PATH, ETC_MODEL_PATH, \
                        SGD_MODEL_PATH, MLP_MODEL_PATH, RF_MODEL_PATH, \
                        PREDICTION_COL_NAME, PREDICTION_COL_NAME_DL1, \
                        PREDICTION_COL_NAME_DL2, PREDICTION_COL_NAME_DL3,\
                        PREDICTION_COL_NAME_DL4, PREDICTION_COL_NAME_XGB, \
                        PREDICTION_COL_NAME_ETC, PREDICTION_COL_NAME_RF, \
                        PREDICTION_COL_NAME_KNN, PREDICTION_COL_NAME_MLP, \
                        PREDICTION_COL_NAME_SVM, PREDICTION_COL_NAME_SGD
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

def probability_features(input_model, input_table, input_mode = "DL"):

    """
    Generate the probability features
    """
    input_table = input_table.fillna(0)
    if input_mode == "DL":
        predict_function = input_model.predict
    elif input_mode == "ML":
        try:
            predict_function = input_model.predict_proba
        except:
            predict_function = input_model.predict
    predicted_prob = predict_function(input_table)
    try:
        if predicted_prob.shape[1] == 2:
            predicted_prob = predicted_prob[:,1]
    except:
        pass
    try:
        table_row = [x[0] for x in predicted_prob.tolist()]
    except:
        table_row = predicted_prob.tolist()
    return table_row

def predict_instances(input_table):

    """
    Load novel instances and use the trained predictors to evaluate their class
    Keep the order as is
    """
    
    MLP_model = probability_features(pickle.load(open(MLP_MODEL_PATH, 'rb')), input_table, input_mode = "ML")
    RF_model = probability_features(pickle.load(open(RF_MODEL_PATH, 'rb')), input_table, input_mode = "ML")
    KNN_model = probability_features(pickle.load(open(KNN_MODEL_PATH, 'rb')), input_table, input_mode = "ML")
    SGD_model = probability_features(pickle.load(open(SGD_MODEL_PATH, 'rb')), input_table, input_mode = "ML")
    XGB_model = probability_features(pickle.load(open(XGB_MODEL_PATH, 'rb')), input_table, input_mode = "ML")
    ETC_model = probability_features(pickle.load(open(ETC_MODEL_PATH, 'rb')), input_table, input_mode = "ML")
    SVM_model = probability_features(pickle.load(open(SVM_MODEL_PATH, 'rb')), input_table, input_mode = "ML")
    
    DL3 = probability_features(load_model(TENSORFLOW_MODEL_PATH_3), input_table, input_mode = "DL")
    DL1 = probability_features(load_model(TENSORFLOW_MODEL_PATH_1), input_table, input_mode = "DL")
    DL4 = probability_features(load_model(TENSORFLOW_MODEL_PATH_4), input_table, input_mode = "DL")
    DL2 = probability_features(load_model(TENSORFLOW_MODEL_PATH_2), input_table, input_mode = "DL")

    ensemble_df = pd.DataFrame({PREDICTION_COL_NAME_MLP: MLP_model, \
                            PREDICTION_COL_NAME_RF: RF_model, \
                            PREDICTION_COL_NAME_KNN: KNN_model, \
                            PREDICTION_COL_NAME_SGD: SGD_model, \
                            PREDICTION_COL_NAME_XGB: XGB_model, \
                            PREDICTION_COL_NAME_ETC: ETC_model, \
                            PREDICTION_COL_NAME_SVM: SVM_model, \
                            PREDICTION_COL_NAME_DL3: DL3, \
                            PREDICTION_COL_NAME_DL1: DL1, \
                            PREDICTION_COL_NAME_DL4: DL4, \
                            PREDICTION_COL_NAME_DL2: DL2})
    DL_ensemble = probability_features(load_model(TENSORFLOW_MODEL_PATH_ENSEMBLE), ensemble_df, input_mode = "DL")
    return pd.DataFrame({PREDICTION_COL_NAME: DL_ensemble,  \
                            PREDICTION_COL_NAME_MLP: MLP_model, \
                            PREDICTION_COL_NAME_RF: RF_model, \
                            PREDICTION_COL_NAME_KNN: KNN_model, \
                            PREDICTION_COL_NAME_SGD: SGD_model, \
                            PREDICTION_COL_NAME_XGB: XGB_model, \
                            PREDICTION_COL_NAME_ETC: ETC_model, \
                            PREDICTION_COL_NAME_SVM: SVM_model, \
                            PREDICTION_COL_NAME_DL3: DL3, \
                            PREDICTION_COL_NAME_DL1: DL1, \
                            PREDICTION_COL_NAME_DL4: DL4, \
                            PREDICTION_COL_NAME_DL2: DL2})

def generate_bin_columns(input_table, target_column_name = PREDICTION_COL_NAME, \
                            new_column_name = "DL Effect"):
    import statistics
    new_column = []
    for index, row in input_table.iterrows():
        average = row[target_column_name]
        if average >= 0.5:
            new_column.append("Synergistic")
        else:
            new_column.append("Non Synergistic")
    input_table[new_column_name] = new_column
    return input_table

input_file = sys.argv[1]

opened_file = pd.read_csv(input_file, sep = SEP, header = 0)

ids = opened_file[["Drug1","Drug2","Cell"]]
features = opened_file.drop(["Drug1","Drug2","Cell"], axis = 1)

predictions_table = predict_instances(features)
final_table = pd.concat([ids,predictions_table], axis = 1)
predictions_table.to_csv("standalone_results/predictions.csv", index = False)

print("===YOUR PREDICTIONS ARE COMPLETE===\n".center(os.get_terminal_size().columns), \
        "===ACCESS THEM AT===\n".center(os.get_terminal_size().columns), \
        "standalone_results/predictions.csv\n".center(os.get_terminal_size().columns))