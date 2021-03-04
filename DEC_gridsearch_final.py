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

SYSTEM_SEP = "/"
CSV_SEP = ","
INTERMEDIATE_SEP = "_"
PARAGRAPH_SEP = "\n"
PARAMETERS_INDEX_FILE = "gridsearch_index_after_grid.csv"
def iterate_dictionary(input_dictionary, target_script = "DEC_synpred_class_final.py", \
						verbose = True):

	dictionary_keys = list(input_dictionary.keys())
	parameters_configuration_list = list(itertools.product(*[input_dictionary[x] for x in dictionary_keys]))
	model_id = 0
	with open(PARAMETERS_INDEX_FILE, "w") as output_file:
		header = "Model ID" + CSV_SEP + CSV_SEP.join(dictionary_keys) + PARAGRAPH_SEP
		output_file.write(header)
		for parameters in parameters_configuration_list:
			model_id += 1
			start_command = "python " + target_script + " "
			for x in parameters:
				start_command +=  '"' + str(x) + '" '
			start_command += str(model_id) + "_after_grid_save"
			os.system(start_command)
			writeable_row = str(model_id) + CSV_SEP + CSV_SEP.join([str(x) for x in parameters]) + PARAGRAPH_SEP
			output_file.write(writeable_row)
			if verbose == True:
				print("Currently evaluating parameter:", parameters)
				print("Parameter set:",model_id)

parameters_dictionary = {"architecture":[[100]*4,[2500]*2,
										[int(4229/2),int(4229/4)], \
										[int(4229/2),int(4229/4),int(4229/16),int(4229/256)]
										],
							"dropout_rate": [0], 
							"dataset": ["PCA","PCA_drop"]}

iterate_dictionary(parameters_dictionary)