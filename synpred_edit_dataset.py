#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract 3 independent datasets depending on cell lines, drug combinations and single drugs, these samples will not appear in either the train or the test subsets.
Write train and test files
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

import os
import pandas as pd
import numpy as np
import sys
from synpred_variables import SYSTEM_SEP, CSV_SEP, PARAGRAPH_SEP, \
							TAB_SEP, INTERMEDIATE_SEP, RANDOM_STATE, \
							SUPPORT_FOLDER, CSV_TERMINATION, REDEPLOYMENT_FOLDER, \
							DATASETS_DICTIONARY
import pickle
import random
import numpy as np

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def write_split_record(input_list, output_file, split_section = ""):

	"""
	Write the names of the split into a csv file for later retrieval
	"""
	with open(SUPPORT_FOLDER + SYSTEM_SEP + split_section + INTERMEDIATE_SEP + output_file + CSV_TERMINATION, "w") as output_file:
		for current_object in input_list:
			output_file.write(current_object + PARAGRAPH_SEP)

def exclude_by_column_value(input_table, target_column, exclude_list, \
								write_mode = False, output_name = "", \
								write_type = "w"):

	"""
	Cut a table according to the values in exclude list and the target column,
	write the table with the subset (if active) and return the subset table
	"""
	shorter_table = input_table[~input_table[target_column].isin(exclude_list)]

	if write_mode == False:
		return shorter_table
	elif write_mode == True:
		exclude_table = input_table[input_table[target_column].isin(exclude_list)].drop(["drug_combinations"], axis = 1)
		if write_type == "w":
			header_mode = True
		elif write_type == "a":
			header_mode = False
		exclude_table.to_csv(output_name, sep = CSV_SEP, index = False, mode = write_type, header = header_mode)
		return shorter_table

def retrieve_split_record(input_file, split_section = ""):

	"""
	Retrieve the written test/train samples to be kept aside on the dataset,
	return a list with the values to drop
	"""
	output_list = []
	with open(SUPPORT_FOLDER + SYSTEM_SEP + split_section + INTERMEDIATE_SEP + input_file + CSV_TERMINATION, "r") as input_file:
		for row in input_file:
			output_list.append(row.replace(PARAGRAPH_SEP,""))
	return output_list

class split_file:

	def __init__(self, mode = "train", input_dataset = "", combination_mode = "only_combinations"):

		"""
		The combination mode and the input dataset define whether or not concentrations are taken into account
		"""
		self.combination_mode = combination_mode
		self.raw_data_file = input_dataset
		self.opened_raw_data = pd.read_csv(self.raw_data_file, sep = CSV_SEP, header = 0)
		self.opened_raw_data[["Drug1","Drug2"]] = self.opened_raw_data[["Drug1","Drug2"]].astype(int).astype(str)

	def split_by_cell(self, number_of_cell_lines = 3, write_output = False):

		"""
		Split the dataset by the number of cell lines
		"""
		self.unique_cells = list(self.opened_raw_data["cell"].unique())
		random.shuffle(self.unique_cells)
		self.unique_cells_test = self.unique_cells[0:number_of_cell_lines]
		self.unique_cells_train = self.unique_cells[number_of_cell_lines:]
		if write_output == True:
			write_split_record(self.unique_cells_test, "cell_lines", split_section = "test")
			write_split_record(self.unique_cells_train, "cell_lines", split_section = "train")

	def split_by_drug_combo(self, number_of_drug_combinations = 5, write_output = False):

		"""
		Split the dataset by the number drug combinations
		"""
		self.combined_column = list(self.opened_raw_data["Drug1"].astype(int).astype(str) + "&" + self.opened_raw_data["Drug2"].astype(int).astype(str))
		random.shuffle(self.combined_column)
		self.unique_drug_combo_test = self.combined_column[0:number_of_drug_combinations]
		self.unique_drug_combo_train = self.combined_column[number_of_drug_combinations:]
		if write_output == True:
			write_split_record(self.unique_drug_combo_test, "drug_combinations", split_section = "test")
			write_split_record(self.unique_drug_combo_train, "drug_combinations", split_section = "train")

	def split_by_drug(self, number_of_drugs = 5, write_output = False):

		"""
		Split the dataset by the number drugs
		"""
		self.unique_drugs = list(self.opened_raw_data["Drug1"].astype(int).astype(str)) + list(self.opened_raw_data["Drug2"].astype(int).astype(str))
		random.shuffle(self.unique_drugs)
		self.unique_drugs_test = self.unique_drugs[0:number_of_drugs]
		self.unique_drugs_train = self.unique_drugs[number_of_drugs:]
		if write_output == True:
			write_split_record(self.unique_drugs_test, "drugs", split_section = "test")
			write_split_record(self.unique_drugs_train, "drugs", split_section = "train")

	def generate_train_dataset(self, train_output_file = "", \
									test_output_file = "", \
									split_test_size = 0.2, \
									cell_file = "", drugs_file = "", \
									drug_combinations_file = ""):

		"""
		Remove the rows containing either the excluded cells, drug combinations or drugs
		"""
		self.opened_raw_data["drug_combinations"] = self.opened_raw_data["Drug1"].astype(int).astype(str) + "&" + self.opened_raw_data["Drug2"].astype(int).astype(str)
		exclude_cells = retrieve_split_record("cell_lines", split_section = "test")
		exclude_drug_combinations = retrieve_split_record("drug_combinations", split_section = "test")
		exclude_drugs = retrieve_split_record("drugs", split_section = "test")

		table_less_cells = exclude_by_column_value(self.opened_raw_data, "cell", exclude_cells, write_mode = True, \
											output_name = cell_file)
		table_less_drug_combinations = exclude_by_column_value(table_less_cells, "drug_combinations", \
											exclude_drug_combinations, write_mode = True, \
											output_name = drug_combinations_file)
		table_less_drugs1 = exclude_by_column_value(table_less_drug_combinations, "Drug1", \
											exclude_drugs, write_mode = True, \
											output_name = drugs_file)
		table_less_drugs2 = exclude_by_column_value(table_less_drugs1, "Drug2", \
											exclude_drugs, write_mode = True, \
											output_name = drugs_file, write_type = "a")
		table_less_drugs2[["Drug1","Drug2"]] = table_less_drugs2[["Drug1","Drug2"]].astype(int).astype(str)
		table_less = table_less_drugs2.drop(["drug_combinations"], axis = 1)

		from sklearn.model_selection import train_test_split

		train_table, test_table = train_test_split(table_less, test_size = split_test_size)
		train_table.to_csv(train_output_file, sep = CSV_SEP, index = False)
		test_table.to_csv(test_output_file, sep = CSV_SEP, index = False)

	def generate_full_agreement(self, full_agreement_column = "full_agreement", \
									class_columns = ["ZIP","HSA","Loewe","Bliss"]):

		"""
		Generate the full-agreement class considering the four metrics and that their threshold is 0
		"""
		def deploy_threshold(input_data, target_columns = class_columns):

			"""
			Convert the column to 0 or one depending on the column values
			"""
			if (input_data[class_columns[0]] > 0) and (input_data[class_columns[1]] > 0) and \
				(input_data[class_columns[2]] > 0) and  (input_data[class_columns[3]] > 0):
				return 1
			else:
				return 0

		def check_agreement(input_data, target_columns = class_columns):

			"""
			Convert the column to 0 or one depending on the column values
			"""
			if (input_data[class_columns[0]] > 0) and (input_data[class_columns[1]] > 0) and \
				(input_data[class_columns[2]] > 0) and  (input_data[class_columns[3]] > 0):
				return 1
			elif (input_data[class_columns[0]] <= 0) and (input_data[class_columns[1]] <= 0) and \
				(input_data[class_columns[2]] <= 0) and  (input_data[class_columns[3]] <= 0):
				return 1
			else:
				return 0

		self.opened_raw_data["full_agreement_val"] = self.opened_raw_data.apply(check_agreement, axis = 1)
		self.opened_raw_data["full_agreement"] = self.opened_raw_data.apply(deploy_threshold, axis = 1)

split_object = split_file(combination_mode = "only_combinations", \
							input_dataset = DATASETS_DICTIONARY["NCI_ALMANAC_classes"])
split_object.generate_full_agreement()
split_object.split_by_cell(write_output = True)
split_object.split_by_drug_combo(write_output = True)
split_object.split_by_drug(write_output = True)
split_object.generate_train_dataset(split_test_size = 0.2, \
							train_output_file = DATASETS_DICTIONARY["train_dataset"], \
							test_output_file = DATASETS_DICTIONARY["test_dataset"], \
							cell_file = DATASETS_DICTIONARY["independent_cell"], \
							drugs_file = DATASETS_DICTIONARY["independent_drugs"], \
							drug_combinations_file = DATASETS_DICTIONARY["independent_drug_combinations"])