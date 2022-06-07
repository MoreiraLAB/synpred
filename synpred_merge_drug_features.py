#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gather the mordred features
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

import pandas as pd
from synpred_variables import SYSTEM_SEP, CSV_SEP, MOLECULES_FOLDER, \
								TXT_TERMINATION, INTERMEDIATE_SEP, RESOURCES_FOLDER, \
								CLASSES_LIST, RANDOM_STATE, \
								MORDRED_RAW_FILE, SUPPORT_FOLDER, \
								MOLECULES_FOLDER, DATASETS_DICTIONARY, \
								MORDRED_RAW_COMBODB_FILE
import sys
from synpred_support_functions import identify_unique_drugs
from synpred_drug_features import drug_features_extractor
import numpy as np
import random

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def retrieve_drugs_features(unique_drugs_list, verbose = True, mode = "smile"):

	"""
	Retrieve the unique drugs features using mordred
	"""
	drugs_features_dictionary = {}
	for current_drug in unique_drugs_list:
		if mode == "combodb_smile":
			current_smile = unique_drugs_list[current_drug]
		elif mode == "smile":
			proper_file = MOLECULES_FOLDER + SYSTEM_SEP + str(int(current_drug)) + TXT_TERMINATION
			current_smile = open(proper_file, "r").readlines()[0]
		try:
			current_features = drug_features_extractor(current_smile, mode = "fetch_only")
			drugs_features_dictionary[current_drug] = current_features
			if verbose == True:
				print("Successfully calculated features for:", current_drug)
		except:
			if verbose == True:
				print("Failed to calculate features for:", current_drug)
			else:
				continue
	return drugs_features_dictionary

def write_features_table(input_features_dictionary, output_file = MORDRED_RAW_FILE):

	"""
	Write a table with the mordred features
	"""

	import pickle
	with open(RESOURCES_FOLDER + SYSTEM_SEP + "drug_features.pkl", "rb") as opened_features:
		mordred_header = pickle.load(opened_features)

	started_features = False
	for current_key in list(input_features_dictionary.keys()):
		#current_mordred_features = pd.DataFrame([[current_key] + input_features_dictionary[current_key]], columns = ["NCI"] + mordred_header)
		current_mordred_features = pd.DataFrame([[current_key] + input_features_dictionary[current_key]], columns = ["SMILE"] + mordred_header)
		if started_features == False:
			output_mordred_table = current_mordred_features
			started_features = True
		elif started_features == True:
			output_mordred_table = pd.concat([output_mordred_table, current_mordred_features], axis = 0)
	output_mordred_table.to_csv(output_file, index = False)

def name_id_dictionary(input_table, drugs_list, drug_1_smile = "drug_1_smile", \
			drug_1_name = "drug_1_name", drug_2_smile = "drug_2_smile", drug_2_name = "drug_2_name"):

	ids_dictionary = {}
	for current_smile in drugs_list:
		current_smile = current_smile.replace("\n","")
		try:
			drug_name = input_table.loc[input_table[drug_1_smile] == current_smile][drug_1_name].iloc[0]
		except:
			drug_name = input_table.loc[input_table[drug_2_smile] == current_smile][drug_2_name].iloc[0]
		ids_dictionary[drug_name] = current_smile
	return ids_dictionary

"""
opened_table = pd.read_csv(DATASETS_DICTIONARY["NCI_ALMANAC"], sep = CSV_SEP, usecols = ["NSC1","NSC2"])
unique_drugs = [x for x in identify_unique_drugs(opened_table, drug_1_col = "NSC1", drug_2_col = "NSC2") if str(x) != "nan"]
features_dictionary = retrieve_drugs_features(unique_drugs, mode = "smile")
write_features_table(features_dictionary)
"""

opened_table = pd.read_csv(DATASETS_DICTIONARY["combodb"], sep = ";", usecols = ["drug_1_smile","drug_1_name","drug_2_smile","drug_2_name"])
unique_drugs = [x for x in identify_unique_drugs(opened_table, drug_1_col = "drug_1_smile", drug_2_col = "drug_2_smile") if str(x) != "nan"]
features_dictionary = retrieve_drugs_features(name_id_dictionary(opened_table, unique_drugs), mode = "combodb_smile")
write_features_table(features_dictionary, output_file = MORDRED_RAW_COMBODB_FILE)