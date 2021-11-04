#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A few helper functions
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

from synpred_variables import CSV_SEP, SYSTEM_SEP, PARAGRAPH_SEP, \
                            INTERMEDIATE_SEP, TAB_SEP, \
                            SCALED_CCLE_START, CCLE_ANNOTATION_FILE, \
                            DROPPABLE_COLUMNS, TARGET_CLASS_COLUMN, \
                            RANDOM_STATE, CONC_DROPPABLE_COLUMNS
import random
import numpy as np
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

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
                        write_mode = True, task_type = "classification"):
    
    """
    Perform model evaluation using a vector of the actual class and the predicted class
    ACTUAL CLASS MUST BE FIRST ARGUMENT
    Also, take attention to change subset_type, to avoid overwritting previous results
    """
    if task_type == "classification":
        from sklearn.metrics import accuracy_score, roc_auc_score, \
                                recall_score, precision_score, f1_score
        from synpred_variables import EVALUATION_DL_FOLDER, CSV_TERMINATION
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
        return [accuracy, precision, recall, auc, f1_value]

    elif task_type == "regression":
        from sklearn.metrics import mean_squared_error, mean_absolute_error, \
                                        r2_score
        import math
        from scipy.stats import pearsonr, spearmanr
        import numpy as np
        from synpred_variables import EVALUATION_DL_FOLDER, CSV_TERMINATION
        if verbose == True:
            print("Currently evaluating:",subset_type, "\n")
        output_file_name = EVALUATION_DL_FOLDER + SYSTEM_SEP + subset_type + CSV_TERMINATION
        try:
            list_input_class = list(input_class.iloc[:,0])
        except:
            list_input_class = list(input_class)
        list_input_predictions = list(input_predictions)
        try:
            RMSE = math.sqrt(mean_squared_error(input_class, input_predictions))
        except:
            RMSE = 10000
        try:
            MSE = mean_squared_error(input_class, input_predictions)
        except:
            MSE = 10000
        try:
            Pearson, pval_Pearson = pearsonr([float(x) for x in list_input_class], [float(x) for x in list_input_predictions])
        except:
            Pearson = -1.0
        try:
            r2 = r2_score(input_class, input_predictions)
        except:
            r2 = -1.0
        try:
            MAE = mean_absolute_error(input_class, input_predictions)
        except:
            MAE = 10000
        try:
            Spearman, pval_Spearman = spearmanr(list_input_class, list_input_predictions)
        except:
            Spearman = -1.0
        if verbose == True:
            print("RMSE:", round(RMSE, 2), "\n",
                   "MSE:" , round(MSE, 2), "\n",
                "Pearson:", round(Pearson, 2), "\n",
                "r^2:", round(r2, 2), "\n",
                "MAE:", round(MAE, 2), "\n",
                "Spearman:", round(Spearman, 2), "\n")
        if write_mode == True:
            with open(output_file_name, "w") as output_file:
                output_file.write("Metric,Value\n")
                output_file.write("RMSE," + str(RMSE) + "\n")
                output_file.write("MSE," + str(MSE) + "\n")
                output_file.write("Pearson," + str(Pearson) + "\n")
                output_file.write("r^2," + str(r2) + "\n")
                output_file.write("MAE," + str(MAE) + "\n")
                output_file.write("Spearman," + str(Spearman) + "\n")
        return [RMSE, MSE, Pearson, r2, MAE, Spearman]

def prepare_dataset(input_train, input_test, drop_columns = DROPPABLE_COLUMNS, \
                    target_column = TARGET_CLASS_COLUMN, subset_size = 0, \
                    sample_size = 1.0, sample_mode = False, task_type = "regression", \
                    final_mode = False, final_reduction = "", final_preprocessing = ""):
    
    """
    Helped function to prepare the dataset for training
    """
    def column_handling(input_subset, columns_list = [], target_column = ""):
        """
        Split the features while removing appropriate columns and yield the target column separately
        """
        current_class_column = input_subset[target_column]
        for current_col in columns_list:
            if current_col in list(input_subset):
                input_subset = input_subset.drop([current_col], axis = 1)
        return input_subset, current_class_column

    import pandas as pd
    if subset_size != 0:
        train_dataset = pd.read_csv(input_train, sep = CSV_SEP, header = 0, nrows = subset_size)
        test_dataset = pd.read_csv(input_test, sep = CSV_SEP, header = 0, nrows = subset_size)
    else:
        train_dataset = pd.read_csv(input_train, sep = CSV_SEP, header = 0)
        test_dataset = pd.read_csv(input_test, sep = CSV_SEP, header = 0)
    if task_type == "classification":
        train_dataset = train_dataset.loc[train_dataset["full_agreement_val"] == 1]
        test_dataset = test_dataset.loc[test_dataset["full_agreement_val"] == 1]
    if sample_mode == True:
        train_dataset = train_dataset.sample(frac = sample_size, axis = 0)
    """
    train_class = train_dataset[target_column]
    for current_col in drop_columns:
        if current_col in list(train_dataset):
            train_dataset = train_dataset.drop([current_col], axis = 1)
    """
    train_features, train_class = column_handling(train_dataset, columns_list = drop_columns, target_column = target_column)
    test_features, test_class = column_handling(test_dataset, columns_list = drop_columns, target_column = target_column)
    """
    test_class = test_dataset[target_column]
    for current_col in drop_columns:
        if current_col in list(test_dataset):
            test_dataset = test_dataset.drop([current_col], axis = 1)
    """
    if final_mode == False:
        return {"train_features": train_features, "train_class": train_class, \
                "test_features": test_features, "test_class": test_class}
    elif final_mode == True:
        from synpred_variables import DATASETS_DICTIONARY
        cell_dataset = pd.read_csv(DATASETS_DICTIONARY[final_reduction + INTERMEDIATE_SEP + \
                            "independent_cell" + INTERMEDIATE_SEP + final_preprocessing], sep = CSV_SEP, header = 0)
        drugs_dataset = pd.read_csv(DATASETS_DICTIONARY[final_reduction + INTERMEDIATE_SEP + \
                            "independent_drugs" + INTERMEDIATE_SEP + final_preprocessing], sep = CSV_SEP, header = 0)
        combo_dataset = pd.read_csv(DATASETS_DICTIONARY[final_reduction + INTERMEDIATE_SEP + \
                            "independent_drug_combinations" + INTERMEDIATE_SEP + final_preprocessing], sep = CSV_SEP, header = 0)
        cell_features, cell_class = column_handling(cell_dataset, columns_list = drop_columns, target_column = target_column)
        drugs_features, drugs_class = column_handling(drugs_dataset, columns_list = drop_columns, target_column = target_column)
        combo_features, combo_class = column_handling(combo_dataset, columns_list = drop_columns, target_column = target_column)
        return {"train_features": train_features, "train_class": train_class, \
                "test_features": test_features, "test_class": test_class, \
                "cell_features": cell_features, "cell_class": cell_class, \
                "drugs_features": drugs_features, "drugs_class": drugs_class, \
                "combo_features": combo_features, "combo_class": combo_class}

def conc_prepare_dataset(input_train = "", input_test = "", drop_columns = CONC_DROPPABLE_COLUMNS, \
                    target_column = TARGET_CLASS_COLUMN, \
                    sample_size = 1.0, sample_fraction_mode = False, task_type = "regression"):
    
    """
    Helper function to prepare the dataset for training
    """
    import pandas as pd
    def generate_class(input_row):
        
        """
        Calculate the class based on the 5 metrics available for concentration
        """
        if (input_row["ZIP"] > 0) and (input_row["Bliss"] > 0) and \
                (input_row["HSA"] > 0) and (input_row["Loewe"] > 0) and \
                (input_row["combo_score"] > 0):
            full_agreement = 1
            full_agreement_val = 1
        elif (input_row["ZIP"] <= 0) and (input_row["Bliss"] <= 0) and \
                (input_row["HSA"] <= 0) and (input_row["Loewe"] <= 0) and \
                (input_row["combo_score"] <= 0):
            full_agreement = 0
            full_agreement_val = 1
        else:
            full_agreement = 0
            full_agreement_val = 0
        return full_agreement, full_agreement_val

    def normalize_concentration(input_table, target_col = ""):

        """
        Apply the normalization to the concentration columns - conc1 and conc2
        """
        from synpred_variables import CONCENTRATION_SCALING_FILE
        opened_concentration_parameters = pd.read_csv(CONCENTRATION_SCALING_FILE, sep = CSV_SEP, header = 0)
        output_col = []
        try:
            current_average = opened_concentration_parameters.loc[opened_concentration_parameters["concentration"] == target_col]["average"][0]
        except:
            current_average = opened_concentration_parameters.loc[opened_concentration_parameters["concentration"] == target_col]["average"][1]
        try:
            current_std = opened_concentration_parameters.loc[opened_concentration_parameters["concentration"] == target_col]["standard_deviation"][0]
        except:
            current_std = opened_concentration_parameters.loc[opened_concentration_parameters["concentration"] == target_col]["standard_deviation"][1]

        output_col = (input_dataframe[target_col] - current_average) / current_std
        return output_col

    if sample_fraction_mode == True:

        import h5py as h5
        import pickle
        from synpred_variables import H5_HEADER_PCA_FILLNA, DATASET_H5_TRAIN_PCA_FILLNA
        with h5.File(DATASET_H5_TRAIN_PCA_FILLNA, "r") as h5_file:
            datasets_list = list(h5_file.keys())
            random.shuffle(datasets_list)
            split_list = datasets_list[0:int(len(datasets_list)*sample_size)]
            input_array = np.concatenate([np.array(h5_file[entry]) for entry in split_list], axis = 0)
            with open(H5_HEADER_PCA_FILLNA,'rb') as header_pickle:
                current_header  = pickle.load(file, encoding='bytes')
            train_dataset = pd.DataFrame(input_array, columns = current_header)
            test_dataset = pd.read_csv(input_test, sep = CSV_SEP, header = 0)

    elif sample_fraction_mode == False:
        train_dataset = pd.read_csv(input_train, sep = CSV_SEP, header = 0)
        test_dataset = pd.read_csv(input_test, sep = CSV_SEP, header = 0)

    train_dataset["conc1"] = normalize_concentration(train_dataset, "conc1")
    train_dataset["conc2"] = normalize_concentration(train_dataset, "conc2")

    test_dataset["conc1"] = normalize_concentration(test_dataset, "conc1")
    test_dataset["conc2"] = normalize_concentration(test_dataset, "conc2")

    if task_type == "classification":
        train_dataset["full_agreement"], train_dataset["full_agreement_val"] = zip(*train_dataset.apply(lambda row: generate_class(row), axis = 1))
        test_dataset["full_agreement"], test_dataset["full_agreement_val"] = zip(*test_dataset.apply(lambda row_test: generate_class(row_test), axis = 1))
        train_dataset = train_dataset.loc[train_dataset["full_agreement_val"] == 1]
        test_dataset = test_dataset.loc[test_dataset["full_agreement_val"] == 1]

    train_class = train_dataset[target_column]
    for current_col in drop_columns:
        if current_col in list(train_dataset):
            train_dataset = train_dataset.drop([current_col], axis = 1)
    
    test_class = test_dataset[target_column]
    for current_col in drop_columns:
        if current_col in list(test_dataset):
            test_dataset = test_dataset.drop([current_col], axis = 1)

    return {"train_features": train_dataset, "train_class": train_class, \
            "test_features": test_dataset, "test_class": test_class}

def identify_unique_drugs(input_table, drug_1_col = "Drug1", drug_2_col = "Drug2"):

    """
    Identify the unique drugs and download smile
    """
    unique_drugs_1 = list(input_table[drug_1_col].unique())
    unique_drugs_2 = list(input_table[drug_2_col].unique())
    return list(set(unique_drugs_1 + unique_drugs_2))

def extract_smile_from_NSC(input_drug):

    """
    Retrieve the SMILE from a sdf object 
    """
    try:
        from pubchempy import Compound, get_compounds
        results = get_compounds("NSC-" + str(int(input_drug)), "name")
        return results[0].isomeric_smiles
        
    except:
        from rpy2.robjects.packages import importr
        import rpy2.robjects as robjects
        rcellminer = importr('rcellminer')
        retrieved_smile = rcellminer.getDrugName([str(int(input_drug))])
        return retrieved_smile[0]