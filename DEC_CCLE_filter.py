#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Treat the CCLE data
conda activate black
tensorflow version 1.15
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

from DEC_variables import CSV_SEP, SYSTEM_SEP, PARAGRAPH_SEP, \
							INTERMEDIATE_SEP, TAB_SEP, \
							SCALED_CCLE_START, H5_DATASET_RAW

from DEC_variables import DEFAULT_LOCATION, CCLE_FOLDER, \
							SUPPORT_FOLDER, CCLE_DATASET_LOG_FILE, \
							REDEPLOYMENT_FOLDER, CCLE_ID_COLUMN_NAME, \
							CCLE_ID_COLUMN_SEP, SUBSET_CCLE

from DEC_variables import H5_DATASET_RAW

import os
import pandas as pd
#pd.set_option('display.max_rows', 100)
#pd.set_option('display.max_columns', 100)


from DEC_support_functions import open_log_file
import sys
import numpy as np
import pickle
import h5py as h5

EXCHANGE_SEPARATORS = {"tab": "\t", "comma": ","}
CCLE_COLUMNS_KEPT_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "columns_kept_names.csv"

def open_CCLE_file(input_file, header_length = 3, split_sep = " "):

	"""
	Open the CCLE_files for initial inspection
	"""
	target_location = CCLE_FOLDER + SYSTEM_SEP + input_file
	sample_table, header = [], []
	written_header = False
	with open(target_location, "r") as CCLE_file:
		for current_row_i in range(header_length):
			current_row = next(CCLE_file).replace(PARAGRAPH_SEP,"").replace(" ","").split(split_sep)
			print(current_row)
			if written_header == False:
				header = current_row
				written_header = True
				continue
			sample_table.append(current_row)

	#sample_dataframe = pd.DataFrame(sample_table, columns = header)
	#print(sample_dataframe)

def edit_ids_column(input_dataframe, ids_list):

	"""
	Aggregate the id columns into a single column and remove the individual columns
	"""
	input_dataframe[CCLE_ID_COLUMN_NAME] = ""
	for current_id in ids_list:
		if current_id == '':
			input_dataframe[CCLE_ID_COLUMN_NAME] = "&" + input_dataframe.iloc[:,0]
			input_dataframe = input_dataframe.drop(["Unnamed: 0"], axis = 1)
		else:
			input_dataframe[CCLE_ID_COLUMN_NAME] = input_dataframe[CCLE_ID_COLUMN_NAME] + "&" + input_dataframe[current_id].astype(str)
			input_dataframe = input_dataframe.drop([current_id], axis = 1)

	return input_dataframe

def dataset_processing(input_dataframe, input_dictionary, file_id, write_scaler = True):

	"""
	Remove dropable columns, normalize dataset and standardize_ids
	"""
	from sklearn.preprocessing import StandardScaler
	from sklearn.feature_selection import VarianceThreshold
	threshold_variance = 0.00
	ids_aggregated_dataframe = edit_ids_column(input_dataframe, input_dictionary["ids_list"])
	
	del input_dataframe

	ids_aggregated_dataframe.fillna(0, inplace=True)
	ids_only_dataframe = ids_aggregated_dataframe[CCLE_ID_COLUMN_NAME]
	dropable_columns = input_dictionary["drop_columns"] + [CCLE_ID_COLUMN_NAME]
	
	if "" in dropable_columns: dropable_columns.remove("")
	no_ids_dataframe = ids_aggregated_dataframe.drop(dropable_columns, axis = 1)

	"""
	Remove missing values and split the ids and the other values
	"""

	no_categorical_dataframe = pd.get_dummies(no_ids_dataframe*1)

	"""
	Remove column with all equal or all different values
	"""
	selector = VarianceThreshold(threshold = threshold_variance)
	selector.fit(no_categorical_dataframe)
	selection_vector = selector.get_support()
	low_variance_dataframe = no_categorical_dataframe.loc[:, selection_vector]
	del no_categorical_dataframe

	"""
	Normalize the data and save the normalizer objects with pickles
	"""
	scaler = StandardScaler()
	scaler.fit(low_variance_dataframe)
	scaled_daframe = pd.DataFrame(scaler.transform(low_variance_dataframe), \
									columns = list(low_variance_dataframe))
	del low_variance_dataframe

	if write_scaler == True:
		scaler_name = REDEPLOYMENT_FOLDER + SYSTEM_SEP + SCALED_CCLE_START + \
						INTERMEDIATE_SEP + file_id + ".pkl"
		pickle.dump(scaler, open(scaler_name, 'wb'))

	return ids_only_dataframe, scaled_daframe

def generate_raw_CCLE(dataset_log_file = CCLE_DATASET_LOG_FILE,  \
						columns_names_file = CCLE_COLUMNS_KEPT_FILE):

	"""
	Generate a raw h5 with unique keys for every entry for every dataset
	"""
	CCLE_files_dictionary = open_log_file(dataset_log_file)
	with open(columns_names_file, "w") as register_file:
		for current_id in CCLE_files_dictionary.keys():
			print("Currently evaluating:",current_id)
			file_location = CCLE_FOLDER + SYSTEM_SEP + CCLE_files_dictionary[current_id]["file_name"]
			current_opened_file = pd.read_csv(file_location, skiprows = int(CCLE_files_dictionary[current_id]["skip_rows"]), \
												header = 0, na_values = ["nan","","     NA","    NaN"],\
												sep = EXCHANGE_SEPARATORS[CCLE_files_dictionary[current_id]["sep"]])
			print(current_opened_file.shape)
			ids_dataframe, processed_dataframe = dataset_processing(current_opened_file, CCLE_files_dictionary[current_id], \
														current_id, write_scaler = True)
			columns_names_row = current_id + CSV_SEP + CCLE_files_dictionary[current_id]["file_name"] + \
								CSV_SEP + CSV_SEP.join([str(x) for x in list(processed_dataframe)]) + PARAGRAPH_SEP
			register_file.write(columns_names_row)
			joint_dataframe = pd.concat([ids_dataframe,processed_dataframe], axis = 1)
			del ids_dataframe, processed_dataframe

			print("Shape:", joint_dataframe.shape)
			output_name = CCLE_FOLDER + SYSTEM_SEP + current_id + INTERMEDIATE_SEP + SUBSET_CCLE
			joint_dataframe.to_csv(output_name, index = False)
			del joint_dataframe
	


generate_raw_CCLE(dataset_log_file = CCLE_DATASET_LOG_FILE)
#open_CCLE_file("10_CCLE_RNAseq_genes_counts_20180929.gct", split_sep = "\t", header_length = 5)