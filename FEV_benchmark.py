#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate CCLE features for benchmark dataset 
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

import pandas as pd
import os
import sys
import pickle
import numpy as np
from DEC_support_functions import open_log_file
from DEC_variables import CCLE_DATASET_LOG_FILE, CCLE_FOLDER, \
                            INTERMEDIATE_SEP, PARAGRAPH_SEP, CSV_SEP, \
                            SYSTEM_SEP, CSV_TERMINATION, REDEPLOYMENT_FOLDER, \
                            SUPPORT_FOLDER

def process_input(input_file = CCLE_DATASET_LOG_FILE, verbose = True):

    opened_index = open_log_file(input_file)
    transpose_list = ["7","12","17","22"]
    
    locate_list = ["BT549_BREAST","CAL851_BREAST","CAL148_BREAST", \
                    "CAL51_BREAST","DU4475_BREAST","HCC1599_BREAST", \
                    "MDAMB436_BREAST","MDAMB231_BREAST","MFM223_BREAST", \
                    "HELA_CERVIX", "HEPG2_LIVER"]
    #locate_list = ['A2058_SKIN', 'A2780_OVARY', 'A375_SKIN', 'A427_LUNG', \
    #                'CAOV3_OVARY', 'DLD1_LARGE_INTESTINE', 'ES2_OVARY', \
    #                'HCT116_LARGE_INTESTINE', 'HT144_SKIN', 'HT29_LARGE_INTESTINE', \
    #                'KPL1_BREAST', 'LOVO_LARGE_INTESTINE', 'MDAMB436_BREAST', \
    #                'NCIH1650_LUNG', 'NCIH2122_LUNG', 'NCIH23_LUNG', \
    #                'NCIH460_LUNG', 'NCIH520_LUNG', 'OCUBM_BREAST', \
    #                'OV90_OVARY', 'NIHOVCAR3_OVARY', 'PA1_OVARY', \
    #                'RKO_LARGE_INTESTINE', 'RPMI7951_SKIN', 'SKMES1_LUNG', \
    #                'SKMEL30_SKIN', 'SKOV3_OVARY', 'SW620_LARGE_INTESTINE', \
    #                'SW837_LARGE_INTESTINE', 'T47D_BREAST', 'UACC62_SKIN', \
    #                'UWB1289_OVARY', 'UWB1289_OVARYBRCA1', 'VCAP_PROSTATE', 'ZR751_BREAST']
    output_dictionary = {}
    for key in opened_index.keys():

        scaler_loc = REDEPLOYMENT_FOLDER + SYSTEM_SEP + "scaler_CCLE_subset_" + key + ".pkl"
        activate_drop = False
        current_dict = opened_index[key]
        file_name = CCLE_FOLDER + SYSTEM_SEP + current_dict["file_name"]
        if current_dict["sep"] == "comma":
            current_sep = ","
        if current_dict["sep"] == "tab":
            current_sep = "\t"
        if len(current_dict["drop_columns"]) > 1:
            activate_drop = True
        else:
            activate_drop = False
        opened_file = pd.read_csv(file_name, skiprows = int(current_dict["skip_rows"]), \
                                    sep = current_sep, na_values=["nan","NaN","NaN","","     NA","    NaN", np.nan,np.NaN], header = 0).fillna(0)
        
        if current_dict["ids_list"][0] != "":
            id_columns = opened_file[current_dict["ids_list"]]
            opened_file = opened_file.drop(current_dict["ids_list"], axis = 1)
        if activate_drop == True:
            opened_file = opened_file.drop(current_dict["drop_columns"], axis = 1)
        
        
        if key == "5":
            index_col = id_columns["CellLineName"]
        elif key == "6":
            index_col = id_columns["CCLE_ID"]
        elif key == "14":
            index_col = opened_file["Unnamed: 0"]
            opened_file = opened_file.drop(["Unnamed: 0"], axis = 1)
        with open(scaler_loc, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
            opened_file = pd.DataFrame(scaler.transform(opened_file))
            
        if key in transpose_list:
            opened_file = opened_file.transpose()
            id_columns = id_columns.transpose()
        if key not in ["5","6","14"]:
            index_col = opened_file.index
        opened_file["index"] = index_col
        opened_file = opened_file.reset_index(drop = True)
        output_table = []
        for current_cell_line in locate_list:
            cell_row = opened_file.loc[opened_file["index"] == current_cell_line]
            
            if cell_row.shape[0] == 0:
                output_table.append([0]*(cell_row.shape[1] - 1) + [current_cell_line])
            else:
                output_table.append(cell_row.values.tolist()[0])
        output_file_name = CCLE_FOLDER + SYSTEM_SEP + key + INTERMEDIATE_SEP + "benchmark_raw" + CSV_TERMINATION
        final_table = pd.DataFrame(output_table)
        final_table.to_csv(output_file_name, index = False)
        if verbose == True:
            print("Written benchmark raw:",  key)

def process_tables(input_file = CCLE_DATASET_LOG_FILE, verbose = True):

    opened_index = open_log_file(input_file)

    transpose_list = ["7","12","17","22"]
    for key in opened_index.keys():
        
        pca_loc = SUPPORT_FOLDER + SYSTEM_SEP + key + "_PCA_transform.pkl"
        raw_loc = CCLE_FOLDER + SYSTEM_SEP + key + INTERMEDIATE_SEP + "benchmark_raw" + CSV_TERMINATION
        output_loc = CCLE_FOLDER + SYSTEM_SEP + key + INTERMEDIATE_SEP + "benchmark_processed" + CSV_TERMINATION

        opened_data = pd.read_csv(raw_loc, sep = CSV_SEP, na_values = ["     NA","    NaN"],header = 0).fillna(0)
        id_column = opened_data.iloc[:,-1]
        opened_data = opened_data.iloc[:, :-1]
            
        
        with open(pca_loc, "rb") as pca_file:
            pca = pickle.load(pca_file)
            reduced_data = pca.transform(opened_data)
        full_df = pd.DataFrame(reduced_data)
        full_df["index"] = id_column
        full_df.to_csv(output_loc, index = False)
        if verbose == True:
            print("Currently pre-processing:", key)
            
process_input()
process_tables()