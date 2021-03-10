#!/usr/bin/env python

"""
Run Standalone synpred version.
You should have downloaded the models and setup the environment and folders as said in the Github page
To run this script, just type, after starting your environment:
python standalone_feature_extraction.py "your_file_name.csv"
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SYNPRED"

import standalone_variables
from standalone_variables import INTERMEDIATE_SEP, SYSTEM_SEP, SEP, \
                                HOME, PROCESSED_TERMINATION, \
                                FEATURES_TERMINATION
from standalone_variables import CELL_LINES_FILE, CELL_LINES_ID, \
                                CELL_LINES_TYPE, CELL_LINES_COLUMN, \
                                CELL_LINES_5, CELL_LINES_6, \
                                CELL_LINES_7, CELL_LINES_12, CELL_LINES_14, \
                                CELL_LINES_17, CELL_LINES_22, HOME
import pandas as pd
import string
import sys
import os
import numpy as np
import pandas as pd
from synpred_variables import SYSTEM_SEP, CSV_SEP, MOLECULES_FOLDER, \
                                RAW_EXAMPLE_TABLE, TXT_TERMINATION, \
                                INTERMEDIATE_SEP
import sys
import pubchempy
from synpred_support_functions import identify_unique_drugs

def download_unique_smiles(input_drugs_list, verbose = True):

    """
    Download unique smiles
    """
    from pubchempy import get_compounds
    for current_drug in input_drugs_list:
        try:
            current_smile = get_compounds(current_drug, 'name')[0].isomeric_smiles
            output_name = MOLECULES_FOLDER + SYSTEM_SEP + \
                            current_drug.replace(" ",INTERMEDIATE_SEP) + TXT_TERMINATION
            with open(output_name, "w") as smile_file:
                smile_file.write(current_smile)
            if verbose == True:
                print("Successfully downloaded:", current_drug)
        except:
            if verbose == True:
                print("Failed to download:", current_drug)
            else:
                continue

def write_drug_features(opened_table, output_file, drugs_dictionary):

    """
    Write the drug features into a file in the standalone_results folder
    """
    drug1_features, drug2_features, count = [], [], 1
    for current_drug_1, current_drug_2 in zip(opened_table["Drug1"], opened_table["Drug2"]):
        try:
            drug1_features.append([current_drug_1] + drugs_dictionary[current_drug_1])
            drug2_features.append([current_drug_2] + drugs_dictionary[current_drug_2])
            print("Combination:",current_drug_1, "&", current_drug_2, \
                    "\n",count, "/", opened_table.shape[0])
        except:
            print("Failed to calculate features for combination:", current_drug_1, "&", current_drug_2)
        count += 1

    drug1_table = pd.DataFrame(drug1_features).add_prefix("Drug1_")
    drug2_table = pd.DataFrame(drug2_features).add_prefix("Drug2_")
    drugs_table = pd.concat([drug1_table, drug2_table], axis = 1)
    drugs_table.to_csv(output_file, index = False)

def retrieve_drugs_features(unique_drugs_list, verbose = True):

    """
    Retrieve the unique drugs features using mordred
    """
    from synpred_drug_features import drug_features_extractor
    drugs_features_dictionary = {}
    for current_drug in unique_drugs_list:
        proper_file = MOLECULES_FOLDER + SYSTEM_SEP + \
                current_drug.replace(" ",INTERMEDIATE_SEP) + TXT_TERMINATION
        try:
            current_smile = open(proper_file, "r").readlines()[0]
            current_features = drug_features_extractor(current_smile)
            drugs_features_dictionary[current_drug] = current_features
            if verbose == True:
                print("Successfully calculated features for:", current_drug)
        except:
            if verbose == True:
                print("Failed to calculate features for:", current_drug)
            else:
                continue
    return drugs_features_dictionary

def merge_cell_features(input_file, output_file):

    """
    Merge the drug features with the CCLE features
    """
    drug_features = pd.read_csv(input_file, sep = ",", header = 0)
    cell_5 = pd.read_csv("CCLE_processed/5_CCLE.csv", sep = ",", header = 0).add_prefix("CCLE_5_").rename(columns = {"CCLE_5_ID":"index"})
    cell_6 = pd.read_csv("CCLE_processed/6_CCLE.csv", sep = ",", header = 0).add_prefix("CCLE_6_").rename(columns = {"CCLE_6_ID":"index"})
    cell_7 = pd.read_csv("CCLE_processed/7_CCLE.csv", sep = ",", header = 0).add_prefix("CCLE_7_").rename(columns = {"CCLE_7_ID":"index"})
    cell_12 = pd.read_csv("CCLE_processed/12_CCLE.csv", sep = ",", header = 0).add_prefix("CCLE_12_").rename(columns = {"CCLE_12_ID":"index"})
    cell_14 = pd.read_csv("CCLE_processed/14_CCLE.csv", sep = ",", header = 0).add_prefix("CCLE_14_").rename(columns = {"CCLE_14_ID":"index"})
    cell_17 = pd.read_csv("CCLE_processed/17_CCLE.csv", sep = ",", header = 0).add_prefix("CCLE_17_").rename(columns = {"CCLE_17_ID":"index"})
    cell_22 = pd.read_csv("CCLE_processed/22_CCLE.csv", sep = ",", header = 0).add_prefix("CCLE_22_").rename(columns = {"CCLE_22_ID":"index"})
    
    med_5 = pd.merge(cell_5, cell_6, how = "left", on = "index")
    id_med_5 = med_5["index"]
    med_5 = med_5.drop(["index"], axis = 1)
    med_5["index"] = id_med_5

    med_6 = pd.merge(med_5, cell_7, how = "left", on = "index")
    id_med_6 = med_6["index"]
    med_6 = med_6.drop(["index"], axis = 1)
    med_6["index"] = id_med_6

    med_7 = pd.merge(med_6, cell_12, how = "left", on = "index")
    id_med_7 = med_7["index"]
    med_7 = med_7.drop(["index"], axis = 1)
    med_7["index"] = id_med_7

    med_12 = pd.merge(med_7, cell_14, how = "left", on = "index")
    id_med_12 = med_12["index"]
    med_12 = med_12.drop(["index"], axis = 1)
    med_12["index"] = id_med_12

    med_14 = pd.merge(med_12, cell_17, how = "left", on = "index")
    id_med_14 = med_14["index"]
    med_14 = med_14.drop(["index"], axis = 1)
    med_14["index"] = id_med_14

    med_17 = pd.merge(med_14, cell_22, how = "left", on = "index")
    id_med_17 = med_17["index"]
    med_17 = med_17.rename(columns = {"index":"Cell"})
    started_merging = False
    for index, row in drug_features.iterrows():
        current_dataframe = pd.concat([row]*med_17.shape[0], axis = 1).transpose()
        current_merged_dataframe = pd.concat([current_dataframe.reset_index(), \
                                    med_17.reset_index()],   
                                    axis = 1).drop(["index"], axis = 1).rename(columns = {"Drug1_0":"Drug1", "Drug2_0":"Drug2"})
        if started_merging == False:
            final_dataframe = current_merged_dataframe
            started_merging = True
        elif started_merging == True:
            final_dataframe = pd.concat([final_dataframe, current_merged_dataframe], axis = 0)

    final_dataframe.to_csv(output_file, index = False)

base_file = sys.argv[1]
opened_file = pd.read_csv(base_file, header = 0, sep = SEP)
unique_drugs_list = identify_unique_drugs(opened_file)
download_unique_smiles(unique_drugs_list)

drug_features_file = "standalone_results/" + base_file.split(".")[0] + "_drug_features.csv"
all_features_file = "standalone_results/" + base_file.split(".")[0] + "_all_features.csv"
drug_features_dictionary = retrieve_drugs_features(unique_drugs_list)
write_drug_features(opened_file, drug_features_file, drug_features_dictionary)
merge_cell_features(drug_features_file, all_features_file)

run_DL_command = "python standalone_deploy_model.py " + HOME + \
                    SYSTEM_SEP + "standalone_results/" + base_file.split(".")[0] + "_all_features.csv"
os.system(run_DL_command)
