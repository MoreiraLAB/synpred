#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gather the mordred features, generate the base dataset and split it in train and test
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

import pandas as pd
from synpred_variables import SYSTEM_SEP, CSV_SEP, MOLECULES_FOLDER, \
								RAW_EXAMPLE_TABLE, TXT_TERMINATION, \
								INTERMEDIATE_SEP
import sys
from synpred_support_functions import identify_unique_drugs
from synpred_drug_features import drug_features_extractor

def retrieve_drugs_features(unique_drugs_list, verbose = True):

	"""
	Retrieve the unique drugs features using mordred
	"""
	drugs_features_dictionary = {}
	for current_drug in unique_drugs_list:
		proper_file = MOLECULES_FOLDER + SYSTEM_SEP + current_drug.replace(" ",INTERMEDIATE_SEP) + TXT_TERMINATION
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

def join_smile_features(input_table, drug_features, \
						drug_1_col = "Drug1", drug_2_col = "Drug2"):

	"""
	Join the SMILE features, calculated with morderted
	"""
	output_table = []
	for index, row in input_table.iterrows():
		drug_1_features = drug_features[row[drug_1_col]]
		drug_2_features = drug_features[row[drug_2_col]]
		current_features = drug_1_features + drug_2_features
		print(current_features)
		sys.exit()



opened_table = pd.read_csv(RAW_EXAMPLE_TABLE, sep = CSV_SEP)
unique_drugs = identify_unique_drugs(opened_table)
features_dictionary = retrieve_drugs_features(unique_drugs)
join_smile_features(opened_table, features_dictionary)