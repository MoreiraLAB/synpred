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
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy.random import seed
import random
import sys
import pickle
from standalone_variables import HOME, SUPPORT_FOLDER, SYSTEM_SEP, SEP, \
                        CELL_LINES_COLUMN, PROCESSED_TERMINATION, \
                        PREDICTION_COL_NAME, MODELS_DICTIONARY, REFERENCE_MODELS_LIST, \
                        ENSEMBLE_MODELS_DICTIONARY
import standalone_feature_extraction
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

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
    current_dictionary, ensemble_dictionary = {}, {}
    for current_reference_model in REFERENCE_MODELS_LIST:
        current_dictionary[current_reference_model] = {}
        for current_model in MODELS_DICTIONARY[current_reference_model]:
            start = current_model.split("/")[-1].split("_")[0]
            if start == "PCA":
                current_mode = "ML"
                current_features = probability_features(pickle.load(open(current_model, "rb")), input_table, input_mode = current_mode)
            else:
                current_mode = "DL"
                current_features = probability_features(load_model(current_model), input_table, input_mode = current_mode)
            current_dictionary[current_reference_model][current_model.split(".")[0]] = current_features
        current_table = pd.DataFrame(current_dictionary[current_reference_model])
        current_array = np.apply_along_axis(replace_outliers, axis = 1, arr = current_table.values)
        ensemble_dictionary[current_reference_model] = pd.DataFrame(current_array, columns = list(current_table))

    DL_ensemble = {}
    for current_ensemble_reference_model in REFERENCE_MODELS_LIST:
        DL_ensemble[current_ensemble_reference_model] = probability_features(load_model(ENSEMBLE_MODELS_DICTIONARY[current_ensemble_reference_model]), \
                        ensemble_dictionary[current_ensemble_reference_model], input_mode = "DL") 
    return pd.DataFrame(DL_ensemble)

def generate_voting_column(input_table):

    """
    Generate a column counting the number of predictors that classified the entry as synergistic 
    """
    voting_column = []
    for index, current_row in input_table[REFERENCE_MODELS_LIST].iterrows():
        current_count = 0
        if current_row["Full-agreement"] >= 0.5:
            current_count += 1
        if current_row["ZIP"] > 0.0:
            current_count += 1
        if current_row["Bliss"] > 0.0:
            current_count += 1
        if current_row["Loewe"] > 0.0:
            current_count += 1
        if current_row["HSA"] > 0.0:
            current_count += 1
        voting_column.append(current_count)

    return voting_column

input_file = sys.argv[1]
opened_file = pd.read_csv(input_file, header = 0, sep = ",")
prediction_loc = input_file.split(".")[0] + "_prediction.csv"

first = True
for index, current_row in opened_file.iterrows():
    try:
        if first == True:
            results_dataframe = standalone_feature_extraction.generate_features_file(current_row["Cell"], current_row["Drug1"], current_row["Drug2"])
            first = False
        elif first == False:
            results_dataframe = pd.concat([results_dataframe, \
                standalone_feature_extraction.generate_features_file(current_row["Cell"], current_row["Drug1"], current_row["Drug2"])], axis = 0)
    except:
        continue

predictions_table = predict_instances(results_dataframe)
final_table = pd.concat([opened_file, predictions_table], axis = 1)
final_table["Synergy Votes"] = generate_voting_column(final_table)
final_table.columns = ["Cell line"] + list(final_table)[1:]
final_table.to_csv(prediction_loc, sep = SEP, index = False)