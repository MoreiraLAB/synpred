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

def generate_analysis_table(target_folder = EVALUATION_DL_FOLDER, mode = "standard", fit_set = "train"):

	"""
	From the input dictionary, generate a results_table
	"""
	written_header = False
	output_table = []
	for files in os.listdir(target_folder):
		split_name = files.split(INTERMEDIATE_SEP)
		ensemble_flag = split_name[0]
		fit_flag =  split_name[-1].split(".")[0]
		id_flag = split_name[1]
		if ensemble_flag != mode or fit_flag != fit_set:
			continue
		file_location = target_folder + SYSTEM_SEP + files
		opened_file = pd.read_csv(file_location, header = 0)
		if written_header == False:
			header = ["Model ID", "File name"] + list(opened_file["Metric"])
			written_header = True
		row = [id_flag, files] + list(opened_file["Value"])
		output_table.append(row)

	output_name = fit_set + INTERMEDIATE_SEP + mode + INTERMEDIATE_SEP + "ensemble_results" + CSV_TERMINATION
	output_dataframe = pd.DataFrame(output_table, columns = header)
	output_dataframe.to_csv(output_name, sep = SECONDARY_CSV_SEP, index = False)
	return output_dataframe

def identify_best(input_table, mode = "standard", fit_set = "train"):

	"""
	Identify the best fit for each model
	"""

	grid_variables = ["Dropout rate","Dataset","Architecture"]
	header = ["ID","File name","Metric","Value"]
	summary_table = []
	for current_metric in METRICS_NAME_LIST:
		subset_table = input_table.loc[input_table[current_metric].idxmax()]
		new_row = [subset_table["Model ID"], subset_table["File name"], current_metric, str(subset_table[current_metric])]
		summary_table.append(new_row)
	output_name = "summary_ensemble" + INTERMEDIATE_SEP + mode + INTERMEDIATE_SEP + fit_set + CSV_TERMINATION
	summary_dataframe = pd.DataFrame(summary_table, columns = header)
	summary_dataframe.to_csv(output_name, sep = SECONDARY_CSV_SEP, index = False)

"""

Generate full grid results table

"""

ensemble_train_table = generate_analysis_table(mode = "ensemble", fit_set = "train")
ensemble_test_table = generate_analysis_table(mode = "ensemble", fit_set = "test")

identify_best(ensemble_train_table, mode = "ensemble", fit_set = "train")
identify_best(ensemble_test_table, mode = "ensemble", fit_set = "test")
