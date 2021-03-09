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
								INTERMEDIATE_SEP, RESOURCES_FOLDER, \
								THRESHOLDS_DICTIONARY, CLASSES_LIST, \
								RANDOM_STATE, TRAIN_DATASET, TEST_DATASET, \
								MORDRED_RAW_FILE, SUPPORT_FOLDER
import sys
from synpred_support_functions import identify_unique_drugs
from synpred_drug_features import drug_features_extractor
import numpy as np
import random

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

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

def write_features_table(input_features_dictionary, output_file = MORDRED_RAW_FILE):

	"""
	Write a table with the mordred features
	"""

	import pickle
	with open(RESOURCES_FOLDER + SYSTEM_SEP + "drug_features.pkl", "rb") as opened_features:
		mordred_header = pickle.load(opened_features)

	started_features = False
	for current_key in list(input_features_dictionary.keys()):
		current_mordred_features = pd.DataFrame([[current_key] + input_features_dictionary[current_key]], columns = ["NCI"] + mordred_header)
		if started_features == False:
			output_mordred_table = current_mordred_features
			started_features = True
		elif started_features == True:
			output_mordred_table = pd.concat([output_mordred_table, current_mordred_features], axis = 0)
	output_mordred_table.to_csv(output_file, index = False)

def join_smile_features(input_table, drug_features, \
						drug_1_col = "Drug1", drug_2_col = "Drug2"):

	"""
	Join the SMILE features, calculated with mordred
	"""
	import pickle
	with open(RESOURCES_FOLDER + SYSTEM_SEP + "drug_features.pkl", "rb") as opened_features:
		mordred_header = pickle.load(opened_features)

	output_table = pd.DataFrame()
	started = False
	for index, row in input_table.iterrows():
		drug_1_features = pd.DataFrame([drug_features[row[drug_1_col]]], columns = mordred_header)
		drug_2_features = pd.DataFrame([drug_features[row[drug_2_col]]], columns = mordred_header).add_suffix(".1")
		current_features = pd.DataFrame([[row["Cell"], row[drug_1_col], row[drug_2_col]]], columns = ["cell","drug1","drug2"])
		current_features = pd.concat([current_features,drug_1_features, drug_2_features], axis = 1)
		for current_class in CLASSES_LIST:
			current_class_value = row[current_class]
			if current_class_value >= THRESHOLDS_DICTIONARY[current_class]:
				current_features[current_class] = 1
			elif current_class_value < THRESHOLDS_DICTIONARY[current_class]:
				current_features[current_class] = 0
		if started == False:
			output_table = current_features
			started = True
		elif started == True:
			output_table = pd.concat([output_table, current_features], axis = 0)
	return output_table

def split_table(input_table, split_ratio = 0.2, \
				output_train = TRAIN_DATASET, \
				output_test = TEST_DATASET):

	"""
	Split output table into train and test subsets
	"""

	from sklearn.model_selection import train_test_split

	train_table, test_table = train_test_split(input_table, test_size = split_ratio)

	train_table.to_csv(output_train, sep = CSV_SEP)
	test_table.to_csv(output_test, sep = CSV_SEP)

opened_table = pd.read_csv(RAW_EXAMPLE_TABLE, sep = CSV_SEP)
unique_drugs = identify_unique_drugs(opened_table)
features_dictionary = retrieve_drugs_features(unique_drugs)
full_table = join_smile_features(opened_table, features_dictionary)
write_features_table(features_dictionary)
split_table(full_table)