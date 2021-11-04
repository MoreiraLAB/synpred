"""
Script to identify the best performing parameters in the gridsearch
"""
import os
import pandas as pd
import sys
from synpred_variables import SYSTEM_SEP, PARAGRAPH_SEP, CSV_SEP, \
                                SUPPORT_FOLDER, DATASETS_DICTIONARY, \
                                DATASETS_DICTIONARY_NO_CONCENTRATION, \
                                INTERMEDIATE_SEP, ML_GRIDSEARCH_RESULTS, \
                                ML_GRIDSEARCH_PARAMETERS, DL_GRIDSEARCH_PARAMETERS, \
                                EVALUATION_DL_FOLDER, DL_GRIDSEARCH_RESULTS

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

def process_ML_parameters(input_file = ML_GRIDSEARCH_RESULTS, \
                            output_file = ML_GRIDSEARCH_RESULTS):
    """
    Open the input csv file and identify the best performing parameters for non-DL models
    """
    results_dictionary = {}
    regression_list = ["Loewe","Bliss","ZIP","HSA"]
    count_regressor = 0
    with open(output_file, "w") as write_file:
        with open(input_file, "r") as read_file:
            writing_mode = False
            header = "Split mode" + CSV_SEP + "Model" + CSV_SEP + "Target" + CSV_SEP + \
                         "parameters" + PARAGRAPH_SEP
            write_file.write(header)
            holder_dataset = ""
            for row in read_file:
                split_row = row.replace(PARAGRAPH_SEP, "").split(CSV_SEP)
                if split_row[-1] not in ["classification", "regression"]:
                    holder_row = split_row
                    split_mode, ML_model, parameters_set = holder_row[0], holder_row[1], holder_row[2:] 
                    register_mode = True

                elif (split_row[-1] in ["classification", "regression"]) and (register_mode == True):
                    register_mode = False
                    if (split_row[-1] == "regression") and (count_regressor == len(regression_list)) and (ML_model == "RF"):
                        count_regressor = 0
                        current_target = regression_list[count_regressor]
                        count_regressor = 1

                    elif (split_row[-1] == "regression") and (count_regressor < len(regression_list)) and (ML_model == "RF"):
                        current_target = regression_list[count_regressor]
                        count_regressor += 1
                    
                    elif (split_row[-1] == "classification"):
                        current_target = "Full-agreement"

                    output_row = split_mode + CSV_SEP + ML_model + CSV_SEP + current_target + CSV_SEP + CSV_SEP.join(parameters_set) + PARAGRAPH_SEP
                    write_file.write(output_row)

def process_DL_parameters(input_file = DL_GRIDSEARCH_RESULTS, \
                            results_folder = EVALUATION_DL_FOLDER, output_file = DL_GRIDSEARCH_PARAMETERS, \
                            starting_file = ""):

    """
    Process the results for the DL Gridsearch results
    """

    opened_file = open(input_file, "r").readlines()
    header = opened_file[0].replace(PARAGRAPH_SEP,"").split(CSV_SEP)
    final_list = []
    for row in opened_file[1:]:
        row = row.replace(PARAGRAPH_SEP, "").split("[")
        current_id = row[0].replace(CSV_SEP,"")
        split_row = row[1].split("]")
        current_architecture = split_row[0].replace(CSV_SEP + " ", "-")
        remaining_cols = split_row[1][1:].split(CSV_SEP)
        final_list.append([int(current_id)] + [current_architecture] + remaining_cols)
    final_dataframe = pd.DataFrame(final_list, columns = header)
    unique_targets = final_dataframe["target"].unique()
    subset_dictionaries = {}
    for entry in unique_targets:
        subset_dictionaries[entry] = list(final_dataframe.loc[final_dataframe["target"] == entry]["Model ID"])

    results_dictionary = {}
    lower_is_best_list = ["RMSE","MSE","MAE"]
    for current_target in subset_dictionaries:
        results_dictionary[current_target] = {"train": {}, "test": {}}
        for current_id in subset_dictionaries[current_target]:
            opened_train = pd.read_csv(results_folder + SYSTEM_SEP + str(current_id) + INTERMEDIATE_SEP + "train.csv", \
                            sep = CSV_SEP, header = 0)
            opened_test = pd.read_csv(results_folder + SYSTEM_SEP + str(current_id) + INTERMEDIATE_SEP + "test.csv", \
                            sep = CSV_SEP, header = 0)
            for index_train, row_train in opened_train.iterrows():
                if row_train["Metric"] not in results_dictionary[current_target]["train"]:
                    results_dictionary[current_target]["train"][row_train["Metric"]] = [row_train["Value"], current_id]
                elif row_train["Metric"] in results_dictionary[current_target]["train"]:
                    if row_train["Metric"] not in lower_is_best_list:
                        if row_train["Value"] > results_dictionary[current_target]["train"][row_train["Metric"]][0]:
                            results_dictionary[current_target]["train"][row_train["Metric"]] = [row_train["Value"], current_id]
                    elif row_train["Metric"] in lower_is_best_list:
                        if abs(row_train["Value"]) < results_dictionary[current_target]["train"][row_train["Metric"]][0]:
                            results_dictionary[current_target]["train"][row_train["Metric"]] = [row_train["Value"], current_id]
            for index_test, row_test in opened_test.iterrows():
                if row_test["Metric"] not in results_dictionary[current_target]["test"]:
                    results_dictionary[current_target]["test"][row_test["Metric"]] = [row_test["Value"], current_id]
                elif row_test["Metric"] in results_dictionary[current_target]["test"]:
                    if row_test["Metric"] not in lower_is_best_list:
                        if row_test["Value"] > results_dictionary[current_target]["test"][row_test["Metric"]][0]:
                            results_dictionary[current_target]["test"][row_test["Metric"]] = [row_test["Value"], current_id]
                    elif row_test["Metric"] in lower_is_best_list:
                        if abs(row_test["Value"]) < results_dictionary[current_target]["test"][row_test["Metric"]][0]:
                            results_dictionary[current_target]["test"][row_test["Metric"]] = [row_test["Value"], current_id]
                            
    with open(output_file, "w") as write_file:
        header = "Target,Model ID,Subset,Metric,Value,Architecture,Dropout Rate,Dataset" + PARAGRAPH_SEP
        write_file.write(header)
        for write_target in results_dictionary:
            for write_metric in results_dictionary[write_target]["train"]:
                train_subset = final_dataframe.loc[final_dataframe["Model ID"] == int(results_dictionary[write_target]["train"][write_metric][1])]
                to_write_train = write_target + CSV_SEP + str(results_dictionary[write_target]["train"][write_metric][1]) + CSV_SEP + "train" + CSV_SEP + \
                    write_metric + CSV_SEP + str(results_dictionary[write_target]["train"][write_metric][0]) + CSV_SEP + \
                    str(train_subset["architecture"].values[0]) + CSV_SEP + str(train_subset["dropout_rate"].values[0]) + CSV_SEP + str(train_subset["dataset"].values[0]) + \
                    PARAGRAPH_SEP
                write_file.write(to_write_train)
            for write_metric in results_dictionary[write_target]["test"]:
                test_subset = final_dataframe.loc[final_dataframe["Model ID"] == int(results_dictionary[write_target]["test"][write_metric][1])]
                to_write_test = write_target + CSV_SEP + str(results_dictionary[write_target]["test"][write_metric][1]) + CSV_SEP + "test" + CSV_SEP +\
                    write_metric + CSV_SEP + str(results_dictionary[write_target]["test"][write_metric][0]) + CSV_SEP + \
                    str(test_subset["architecture"].values[0]) + CSV_SEP + str(test_subset["dropout_rate"].values[0]) + CSV_SEP + str(test_subset["dataset"].values[0]) + \
                    PARAGRAPH_SEP
                write_file.write(to_write_test)

process_ML_parameters()
process_DL_parameters()