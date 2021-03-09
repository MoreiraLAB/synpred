#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Join the features with the previously split dataset
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

from synpred_variables import CSV_SEP, SYSTEM_SEP, PARAGRAPH_SEP, \
							INTERMEDIATE_SEP, TAB_SEP, \
							SCALED_CCLE_START, H5_DATASET_RAW

from synpred_variables import DEFAULT_LOCATION, CCLE_FOLDER, \
							SUPPORT_FOLDER, CCLE_DATASET_LOG_FILE, \
							REDEPLOYMENT_FOLDER, CCLE_ID_COLUMN_NAME, \
							TRAIN_DATASET, TEST_DATASET, CCLE_ID_COLUMN_SEP, \
							SUBSET_CCLE, FILTERED_CCLE, CCLE_ANNOTATION_FILE, \
							NCI_ALMANAC_DATASET, NCI_ALMANAC_CELL_COLUMN, \
							CORRECTION_DICTIONARY

from synpred_support_functions import alternative_ID_file

import os
import pandas as pd
#pd.set_option('display.max_rows', 100)
#pd.set_option('display.max_columns', 100)

import sys
import numpy as np
import pickle
import h5py as h5

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

def filter_out(target_file, input_files_list):

	"""
	Subset the CCLE large datasets by comparing the ids to the dataset
	"""
	import difflib
	CORRECT_DICTIONARY = {}
	opened_target = pd.read_csv(target_file, header = 0, usecols = [NCI_ALMANAC_CELL_COLUMN])[NCI_ALMANAC_CELL_COLUMN].unique().tolist()
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

unique_CCLE = locate_unique_CCLE()
filter_out(NCI_ALMANAC_DATASET, unique_CCLE)
