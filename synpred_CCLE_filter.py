#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Treat the CCLE data
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

from synpred_variables import CSV_SEP, SYSTEM_SEP, PARAGRAPH_SEP, \
							INTERMEDIATE_SEP, TAB_SEP, SCALED_CCLE_START

from synpred_variables import DEFAULT_LOCATION, CCLE_FOLDER, \
							SUPPORT_FOLDER, CCLE_DATASET_LOG_FILE, \
							REDEPLOYMENT_FOLDER, CCLE_ID_COLUMN_NAME, \
							CCLE_ID_COLUMN_SEP, SUBSET_CCLE, FILTERED_CCLE, \
							CCLE_ANNOTATION_FILE, DATASETS_DICTIONARY, NCI_ALMANAC_CELL_COLUMN, \
							CORRECTION_DICTIONARY
import os
import pandas as pd
from synpred_support_functions import open_log_file, alternative_ID_file
import sys
import numpy as np
import pickle

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
	Remove dropable columns by threshold
	"""
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

	return ids_only_dataframe, low_variance_dataframe

def generate_raw_CCLE(dataset_log_file = CCLE_DATASET_LOG_FILE,  \
						columns_names_file = CCLE_COLUMNS_KEPT_FILE):

	"""
	Generate a an output file for every entry for every dataset
	"""
	CCLE_files_dictionary = open_log_file(dataset_log_file)
	with open(columns_names_file, "w") as register_file:
		for current_id in CCLE_files_dictionary.keys():
			print("Currently evaluating:",current_id)
			file_location = CCLE_FOLDER + SYSTEM_SEP + CCLE_files_dictionary[current_id]["file_name"]
			current_opened_file = pd.read_csv(file_location, skiprows = int(CCLE_files_dictionary[current_id]["skip_rows"]), \
												header = 0, na_values = ["nan","","     NA","    NaN"],\
												sep = EXCHANGE_SEPARATORS[CCLE_files_dictionary[current_id]["sep"]])
			print("--Initial shape:", current_opened_file.shape)
			ids_dataframe, processed_dataframe = dataset_processing(current_opened_file, CCLE_files_dictionary[current_id], \
														current_id, write_scaler = True)
			columns_names_row = current_id + CSV_SEP + CCLE_files_dictionary[current_id]["file_name"] + \
								CSV_SEP + CSV_SEP.join([str(x) for x in list(processed_dataframe)]) + PARAGRAPH_SEP
			register_file.write(columns_names_row)
			joint_dataframe = pd.concat([ids_dataframe,processed_dataframe], axis = 1)
			del ids_dataframe, processed_dataframe

			print("--Final shape:", joint_dataframe.shape)
			output_name = CCLE_FOLDER + SYSTEM_SEP + current_id + INTERMEDIATE_SEP + SUBSET_CCLE
			joint_dataframe.to_csv(output_name, index = False)
			del joint_dataframe

def locate_unique_CCLE(input_file = CCLE_DATASET_LOG_FILE):

	"""
	Isolate the IDs for the unique CCLE files 
	"""
	opened_file = open(input_file, "r").readlines()
	unique_ids = []
	for row in opened_file[1:]:
		unique_ids.append(row.split(CSV_SEP)[0])
	return unique_ids

def extend_list(input_list, extend_dictionary = alternative_ID_file()):

	"""
	Add to the original list alternative IDs
	"""

	output_list = []
	for x in input_list:
		if x in list(CORRECTION_DICTIONARY.keys()):
			output_list.append(CORRECTION_DICTIONARY[x])
		if x[0:6] == "SF-539":
			x = "SF-539"
		output_list.append(x)
		x_ammended = x.replace("-","").replace(" ","")

		split_ammended = x_ammended.split("/")
		if len(split_ammended) == 2:
			if split_ammended[0] != "NCI":
				x_ammended = split_ammended[0]
			elif split_ammended[0] == "NCI":
				x_ammended = split_ammended[1]

		split_ammended_2 = x_ammended.split("(")
		if len(split_ammended_2) == 2:
			x_ammended = split_ammended_2[0]

		if x in list(extend_dictionary.keys()):
			for y in extend_dictionary[x]:
				output_list.append(y)
		output_list.append(x_ammended)
	return output_list

def filter_out(target_file, input_files_list, target_column = NCI_ALMANAC_CELL_COLUMN, sep = CSV_SEP):

	"""
	Subset the CCLE large datasets by comparing the ids to the dataset
	"""
	opened_target = pd.read_csv(target_file, header = 0, sep = sep, usecols = [target_column])[target_column].unique().tolist()
	extended_id_list = extend_list(opened_target)
	for current_subset in input_files_list:
		file_location = CCLE_FOLDER + SYSTEM_SEP + current_subset  + INTERMEDIATE_SEP + SUBSET_CCLE
		opened_column = pd.read_csv(file_location, header = 0, usecols = [CCLE_ID_COLUMN_NAME])[CCLE_ID_COLUMN_NAME].tolist()
		drop_rows, count, row_wise = [], 0, True
		detected = []
		for current_id in opened_column:
			split_id = current_id.split(CCLE_ID_COLUMN_SEP)
			exists = False
			for x in split_id:
				if x in extended_id_list:
					exists = True
					detected.append(x)
					break
			if exists == False:
				drop_rows.append(count)
			count += 1
		if len(drop_rows) == len(opened_column):
			row_wise = False
			opened_file = pd.read_csv(file_location, header = 0, nrows = 1)
			drop_columns = []
			for current_column in list(opened_file):
				if current_column in extended_id_list:
					continue
				elif current_column not in extended_id_list:
					drop_columns.append(current_column)
		if row_wise == True:
			final_file = pd.read_csv(file_location, header = 0).drop(drop_rows, axis = 0)
			output_name = CCLE_FOLDER + SYSTEM_SEP + current_subset + INTERMEDIATE_SEP + FILTERED_CCLE
			if final_file.shape[0] != 0:
				final_file.to_csv(output_name, index = False)
			print("Evaluated:",output_name,"with shape:",final_file.shape)
		elif row_wise == False:
			final_file = pd.read_csv(file_location, header = 0).drop(drop_columns, axis = 1)
			output_name = CCLE_FOLDER + SYSTEM_SEP + current_subset + INTERMEDIATE_SEP + FILTERED_CCLE
			if final_file.shape[1] != 0:
				final_file.transpose().to_csv(output_name, index = True)
			print("Evaluated:",output_name,"with shape:",final_file.transpose().shape)

generate_raw_CCLE(dataset_log_file = CCLE_DATASET_LOG_FILE)
unique_CCLE = locate_unique_CCLE()
#filter_out(DATASETS_DICTIONARY["NCI_ALMANAC"], unique_CCLE, )
filter_out(DATASETS_DICTIONARY["combodb"], unique_CCLE, target_column = "cell_line_name", sep = ";" )