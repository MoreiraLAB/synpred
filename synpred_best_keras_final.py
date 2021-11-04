#!/usr/bin/env python

"""
Deploy DL pipeline on the SynPred dataset
with gridsearch
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

import os
import sys
import itertools
from synpred_variables import SYSTEM_SEP, CSV_SEP, \
						INTERMEDIATE_SEP, PARAGRAPH_SEP, \
						SUPPORT_FOLDER, CSV_TERMINATION, \
						DL_GRIDSEARCH_PARAMETERS, BEST_PARAMETERS_INDEX_FILE

def iterate_dictionary(input_meta_dictionary, target_script = "synpred_keras_final.py", \
						verbose = True):
	
	model_id = 0
	with open(BEST_PARAMETERS_INDEX_FILE, "w") as output_file:
		for target in input_meta_dictionary:
			dictionary_keys = list(input_meta_dictionary[target].keys())
			parameters_configuration_list = list(itertools.product(*[input_meta_dictionary[target][x] for x in dictionary_keys]))
			header = "Model ID" + CSV_SEP + CSV_SEP.join(dictionary_keys) + PARAGRAPH_SEP
			output_file.write(header)
			for parameters in parameters_configuration_list:
				model_id += 1
				start_command = "python " + target_script + " "
				for x in parameters:
					start_command +=  '"' + str(x) + '" '
				start_command += str(model_id) + "_after_grid_save"
				os.system(start_command)
				writeable_row = str(target) + CSV_SEP + str(model_id) + CSV_SEP + CSV_SEP.join([str(x) for x in parameters]) + PARAGRAPH_SEP
				output_file.write(writeable_row)
				if verbose == True:
					print("Currently evaluating parameter:", parameters)
					print("Parameter set:",model_id)

def locate_best_parameters(input_file = DL_GRIDSEARCH_PARAMETERS, target_col = "Target", \
							usable_dataset = "PCA_fillna"):

	"""
	Yield the the unique best architectures for each configuration
	"""
	import pandas as pd
	opened_file = pd.read_csv(input_file, sep = CSV_SEP, header = 0)
	unique_targets = list(opened_file[target_col].unique())
	output_dictionary = {}
	for current_target in unique_targets:
		current_subset = opened_file.loc[opened_file[target_col] == current_target]
		best_architectures = [[int(y) for y in x.split("-")] for x in list(current_subset["Architecture"].unique())]
		dropout_rate = current_subset["Dropout Rate"].value_counts().idxmax()
		output_dictionary[current_target] = {"architecture": best_architectures, \
											"dropout_rate": [dropout_rate], \
											"dataset": [usable_dataset], \
											"target": [current_target]}
	return output_dictionary

parameters_after_gridsearch = locate_best_parameters()
iterate_dictionary(parameters_after_gridsearch)