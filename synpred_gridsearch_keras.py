#!/usr/bin/env python

"""
Deploy DL pipeline on the SynPred dataset
with custom made gridsearch
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

import os
import sys
import itertools
import os.path
from synpred_variables import SYSTEM_SEP, CSV_SEP, \
								INTERMEDIATE_SEP, PARAGRAPH_SEP, \
								SUPPORT_FOLDER, CSV_TERMINATION, \
								EVALUATION_DL_FOLDER, DL_GRIDSEARCH_RESULTS, \
								DL_COMBODB_GRIDSEARCH_RESULTS

def iterate_dictionary(input_dictionary, target_script = "synpred_keras.py", \
						verbose = True, gridsearch_output = DL_GRIDSEARCH_RESULTS, \
						other_info = ""):
	
	"""
	Iterate over the dictionary of parameters defined below the function and train all the combinations with keras
	"""
	dictionary_keys = list(input_dictionary.keys())
	parameters_configuration_list = list(itertools.product(*[input_dictionary[x] for x in dictionary_keys]))
	model_id = 0
	with open(gridsearch_output, "w") as output_file:
		header = "Model ID" + CSV_SEP + CSV_SEP.join(dictionary_keys) + PARAGRAPH_SEP
		output_file.write(header)
		for parameters in parameters_configuration_list:
			model_id += 1
			start_command = "python " + target_script + " "
			for x in parameters:
				start_command +=  '"' + str(x) + '" '
			start_command += str(model_id)
			train_path = EVALUATION_DL_FOLDER + SYSTEM_SEP + str(model_id) + INTERMEDIATE_SEP  + "train" + CSV_TERMINATION
			test_path = EVALUATION_DL_FOLDER + SYSTEM_SEP + str(model_id) + INTERMEDIATE_SEP + "test" + CSV_TERMINATION
			if (not os.path.isfile(train_path)) or (not os.path.isfile(test_path)):
				os.system(start_command)
				
			writeable_row = str(model_id) + CSV_SEP + CSV_SEP.join([str(x) for x in parameters]) + PARAGRAPH_SEP
			output_file.write(writeable_row)
			if verbose == True:
				print("Currently evaluating parameter:", parameters)
				print("Parameter set:", model_id)

parameters_dictionary = {"architecture":[[100]*3,[100]*4,\
										[500]*3,[500]*4,\
										[1000]*3,[1000]*4,\
										[2500]*3,[2500]*4,\
										[int(1347/2),int(1347/4),int(1347/16)], \
										[int(1347/2),int(1347/4),int(1347/16),int(1347/256)], \
										[int(4229/2),int(4229/4),int(4229/16)], \
										[int(4229/2),int(4229/4),int(4229/16),int(4229/256)]
										],
							"dropout_rate": [0,0.25,0.5,0.75], 
							"dataset": ["combodb_PCA_fillna","combodb_PCA_dropna","combodb_autoencoder_fillna","combodb_autoencoder_dropna"],
							"target": ["full_agreement","Loewe","Bliss","HSA","ZIP","CSS-RI"]}

iterate_dictionary(parameters_dictionary, gridsearch_output = DL_COMBODB_GRIDSEARCH_RESULTS, other_info = "combodb_")