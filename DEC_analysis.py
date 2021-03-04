#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variables for the Synpred code
source activate black
tensorflow version 1.15
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

import os
import pandas as pd
import sys
from DEC_variables import EVALUATION_NON_DL_FOLDER, EVALUATION_DL_FOLDER, \
							EVALUATION_DL_INDEX, EVALUATION_DL_CUSTOM_INDEX, \
							CSV_SEP, SYSTEM_SEP, PARAGRAPH_SEP, SECONDARY_CSV_SEP, \
							INTERMEDIATE_SEP, CSV_TERMINATION, METRICS_NAME_LIST


def open_index_file(input_file):

	"""
	Open the analysis index file
	Output a dictionary with model id keys and value of
	another dictionary with corresponding dropout rate, dataset
	and architecture
	"""
	opened_raw_file = open(input_file, "r").readlines()
	output_dictionary = {}
	for row in opened_raw_file[1:]:
		split_row = row.replace(PARAGRAPH_SEP, "").split(CSV_SEP)
		model_id = split_row[0]
		dropout_rate = split_row[-2]
		dataset = split_row[-1]
		architecture = CSV_SEP.join([x.replace("[","").replace("]","").replace(" ","") for x in split_row[1:-2]])
		print(architecture)
		output_dictionary[model_id] = {"dropout": dropout_rate, "dataset": dataset, "architecture": architecture}
	return output_dictionary

def generate_analysis_table(input_dictionary, mode = "standard", fit_set = "train"):

	"""
	From the input dictionary, generate a results_table
	"""
	written_header = False
	output_table, count = [], 1
	for model_id_key in input_dictionary.keys():
		if mode == "custom":
			prefix = INTERMEDIATE_SEP + "custom"
		else:
			prefix = ""
		train_location = EVALUATION_DL_FOLDER + SYSTEM_SEP + model_id_key + \
			prefix + INTERMEDIATE_SEP + fit_set + CSV_TERMINATION
		opened_train = pd.read_csv(train_location, header = 0)
		if written_header == False:
			header = ["Model ID", "Dropout rate","Dataset","Architecture"] + list(opened_train["Metric"])
			written_header = True
		row = [model_id_key, input_dictionary[model_id_key]["dropout"], input_dictionary[model_id_key]["dataset"], \
				input_dictionary[model_id_key]["architecture"]] + list(opened_train["Value"])
		output_table.append(row)
		if count % 10 == 0:
			print("Current evaluating:", count, "/", len(list(input_dictionary.keys())))
		count += 1
	output_name = fit_set + INTERMEDIATE_SEP + mode + INTERMEDIATE_SEP + "grid_results" + prefix + CSV_TERMINATION
	output_dataframe = pd.DataFrame(output_table, columns = header)
	output_dataframe.to_csv(output_name, sep = SECONDARY_CSV_SEP, index = False)
	return output_dataframe

def identify_best(input_table, mode = "standard", fit_set = "train"):

	"""
	Identify the best fit for each model
	"""
	
	grid_variables = ["Dropout rate","Dataset","Architecture"]
	header = ["Variable Under Scope","Value Under Scope"] + METRICS_NAME_LIST
	summary_table = []
	for current_variable in grid_variables:
		unique_values = input_table[current_variable].unique()
		for current_unique in unique_values:
			subset_table = input_table.loc[input_table[current_variable] == current_unique].mean().drop("Model ID")
			try:
				subset_table = subset_table.drop(current_variable)
			except:
				pass
			new_row = [current_variable, current_unique] + list(subset_table)
			summary_table.append(new_row)
	output_name = "summary_grid" + INTERMEDIATE_SEP + mode + INTERMEDIATE_SEP + fit_set + CSV_TERMINATION
	summary_dataframe = pd.DataFrame(summary_table, columns = header)
	summary_dataframe.to_csv(output_name, sep = SECONDARY_CSV_SEP, index = False)

"""

Generate full grid results table

"""
current_dictionary_standard = open_index_file(EVALUATION_DL_INDEX)

standard_train_table = generate_analysis_table(current_dictionary_standard, mode = "standard", fit_set = "train")
standard_test_table = generate_analysis_table(current_dictionary_standard, mode = "standard", fit_set = "test")

current_dictionary_custom = open_index_file(EVALUATION_DL_CUSTOM_INDEX)

custom_train_table = generate_analysis_table(current_dictionary_custom, mode = "custom", fit_set = "train")
custom_test_table = generate_analysis_table(current_dictionary_custom, mode = "custom", fit_set = "test")

identify_best(standard_train_table, mode = "standard", fit_set = "train")
identify_best(standard_test_table, mode = "standard", fit_set = "test")

identify_best(custom_train_table, mode = "custom", fit_set = "train")
identify_best(custom_test_table, mode = "custom", fit_set = "test")