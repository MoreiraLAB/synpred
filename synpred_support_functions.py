#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A few helper functions
conda activate synpred
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

from synpred_variables import CSV_SEP, SYSTEM_SEP, PARAGRAPH_SEP, \
                            INTERMEDIATE_SEP, TAB_SEP, \
                            SCALED_CCLE_START, CCLE_ANNOTATION_FILE, \
                            DROPPABLE_COLUMNS, TARGET_CLASS_COLUMN

def open_log_file(input_file_name):

    """
    Open the file with the CCLE datasets information
    """
    opened_file = open(input_file_name, "r").readlines()
    output_dictionary = {}
    for row in opened_file[1:]:
        row = row.replace(PARAGRAPH_SEP,"")
        base_info, ids, dropable_columns = row.split(";")
        file_id, separator, file_name, rows_to_skip, data_type = base_info.split(CSV_SEP)
        ids_list = ids.split(CSV_SEP)
        dropable_columns_list = dropable_columns.split(CSV_SEP)
        output_dictionary[file_id] = {"file": file_id, "sep": separator, \
                                        "file_name": file_name, "skip_rows": rows_to_skip, \
                                        "type": data_type, "ids_list": ids_list, \
                                         "drop_columns": dropable_columns_list}

    return output_dictionary

def alternative_ID_file(input_file = CCLE_ANNOTATION_FILE):

    """
    Open the CCLE annotation file and locate alternative IDs.
    """
    opened_file = open(input_file, "r").readlines()
    output_dictionary = {}
    for row in opened_file[1:]:
        ids_row = []
        split_row = row.replace(PARAGRAPH_SEP, "").split(TAB_SEP)
        first_id = split_row[0].split(INTERMEDIATE_SEP)
        ids_row.append(first_id[0])
        ids_row.append(INTERMEDIATE_SEP.join(first_id[1:]))
        ids_row += split_row[0:3]
        id_ammended = split_row[2].replace("-","").replace(" ","")
        split_clean = id_ammended.split("/")
        if len(split_clean) == 2:
            if split_clean[0] != "NCI":
                id_ammended = split_clean[0]
            elif split_clean[0] == "NCI":
                id_ammended = split_clean[1]

        split_clean_2 = id_ammended.split("(")
        if len(split_clean_2) == 2:
            id_ammended = split_clean_2[0]
        output_dictionary[split_row[2]] = ids_row + [id_ammended]
    return output_dictionary

def model_evaluation(input_class, input_predictions, \
                        subset_type = "test", verbose = False, \
                        write_mode = True):
    
    """
    Perform model evaluation using a vector of the actual class and the predicted class
    ACTUAL CLASS MUST BE FIRST ARGUMENT
    Also, take attention to change subset_type, to avoid overwritting previous results
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, \
                            recall_score, precision_score, f1_score
    from DEC_variables import EVALUATION_DL_FOLDER, CSV_TERMINATION
    if verbose == True:
        print("Currently evaluating:",subset_type, "\n")
    output_file_name = EVALUATION_DL_FOLDER + SYSTEM_SEP + subset_type + CSV_TERMINATION
    try:
        accuracy = accuracy_score(input_class, input_predictions)
    except:
        accuracy = 0.0
    try:
        precision = precision_score(input_class, input_predictions)
    except:
        precision = 0.0
    try:
        recall = recall_score(input_class, input_predictions)
    except:
        recall = 0.0
    try:
        auc = roc_auc_score(input_class, input_predictions)
    except:
        auc = 0.0
    try:
        f1_value = f1_score(input_class, input_predictions)
    except:
        f1_value = 0.0
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
    
    import pandas as pd
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

def identify_unique_drugs(input_table, drug_1_col = "Drug1", drug_2_col = "Drug2"):

    """
    Identify the unique drugs and download smile
    """
    unique_drugs_1 = list(input_table[drug_1_col].unique())
    unique_drugs_2 = list(input_table[drug_2_col].unique())
    return list(set(unique_drugs_1 + unique_drugs_2))