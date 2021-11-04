#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Isolate the drug-drug-cell combinations and match them with their corresponding concentration zero values.
Output the appropriate file format into a csv "processed_*.csv"
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

import pandas as pd
import sys
from synpred_variables import DATASET_FOLDER, SYSTEM_SEP, \
								INTERMEDIATE_SEP, DATASETS_DICTIONARY

def correct_units(input_table, mode = "double"):

	"""
	Convert concentration units to nM
	"""
	if input_table["CONCUNIT1"].unique() == "M":
		input_table["CONC1"] *= 10**9
		input_table["CONCUNIT1"] = "nM"
	if mode == "single":
		return input_table
	if input_table["CONCUNIT2"].unique() == "M":
		input_table["CONC2"] *= 10**9
		input_table["CONCUNIT2"] = "nM"
	return input_table

def locate_concentration_zero(input_table, nan_col = "", cols_to_use = [], \
								left_columns = [], enable_comboscore = False, \
								output_cols_zeros = ["drug_name","concentration","response","unit","cell"]):

	"""
	Generate a table with the concentration zero rows for each drug
	"""
	opened_table = pd.read_csv(input_table, sep = ",", header = 0, usecols = cols_to_use)
	if enable_comboscore == True:
		zeros_table = correct_units(opened_table[opened_table[nan_col].isnull()][left_columns], mode = "single").fillna(0)
	elif enable_comboscore == False:
		zeros_table = correct_units(opened_table[opened_table[nan_col].isnull()][left_columns], mode = "single")
	zeros_table.columns = output_cols_zeros
	combinations_table = opened_table[opened_table[nan_col].notna()]
	return combinations_table, zeros_table

def average_zeros_table(input_table, usable_concentrations, concentration_column = "concentration", \
		response_column = "response"):

	"""
	When there are multiple possible zero concentrations, average these values
	"""
	import numpy as np
	unique_values = list(set(input_table[concentration_column].unique()) & set([x for x in list(usable_concentrations)]))
	output_dataframe = pd.DataFrame(columns = list(input_table))
	for current_unique in unique_values:
		subset_table = input_table.loc[input_table[concentration_column] == current_unique]
		response = subset_table[response_column].mean()
		current_row = subset_table.iloc[0,:]
		current_row["response"] = response
		current_row = pd.DataFrame(current_row.values.reshape(1,-1), columns = list(input_table))
		current_row = current_row[list(input_table)]
		output_dataframe = pd.concat([output_dataframe, current_row], axis = 0)
	return output_dataframe

def fetch_zeros(input_zeros_table, input_subset_table, input_drug_1, \
					input_drug_2, input_cell, final_cols = [], enable_comboscore = False):

	"""
	Access the table with the zeros and generate complete rows for the input drugs
	- Only keeps the concentrations in the combinations provided
	"""

	drug_1_concentration_unique = input_subset_table["conc_r"].unique()
	drug_1_zeros_table = average_zeros_table(input_zeros_table.loc[(input_zeros_table["drug_name"] == int(input_drug_1)) & (input_zeros_table["cell"] == input_cell)], \
									drug_1_concentration_unique)

	drug_2_concentration_unique = input_subset_table["conc_C"].unique()
	drug_2_zeros_table = average_zeros_table(input_zeros_table.loc[(input_zeros_table["drug_name"] == int(input_drug_2)) & (input_zeros_table["cell"] == input_cell)], \
									drug_2_concentration_unique)

	zero_table_1 = pd.concat([drug_1_zeros_table["drug_name"].reset_index(drop = True), \
								pd.DataFrame([input_drug_2]*drug_1_zeros_table.shape[0]), \
								drug_1_zeros_table["concentration"].reset_index(drop = True), \
								pd.DataFrame([0]*drug_1_zeros_table.shape[0]), \
								drug_1_zeros_table["response"].reset_index(drop = True), \
								pd.DataFrame(["nM"]*drug_1_zeros_table.shape[0]), \
								pd.DataFrame(["nM"]*drug_1_zeros_table.shape[0]), \
								drug_1_zeros_table["cell"].reset_index(drop = True)], axis = 1)
	if enable_comboscore == True:
		zero_table_1 = pd.concat([zero_table_1, pd.DataFrame([0]*drug_1_zeros_table.shape[0])], axis = 1)
	zero_table_1.columns = final_cols

	zero_table_2 = pd.concat([pd.DataFrame([input_drug_1]*drug_2_zeros_table.shape[0]), \
								drug_2_zeros_table["drug_name"].reset_index(drop = True), \
								pd.DataFrame([0]*drug_2_zeros_table.shape[0]), \
								drug_2_zeros_table["concentration"].reset_index(drop = True), \
								drug_2_zeros_table["response"].reset_index(drop = True), \
								pd.DataFrame(["nM"]*drug_2_zeros_table.shape[0]), \
								pd.DataFrame(["nM"]*drug_2_zeros_table.shape[0]), \
								drug_2_zeros_table["cell"].reset_index(drop = True)], axis = 1)
	if enable_comboscore == True:
		zero_table_2 = pd.concat([zero_table_2, pd.DataFrame([0]*drug_2_zeros_table.shape[0])], axis = 1)
	zero_table_2.columns = final_cols

	return pd.concat([zero_table_1, zero_table_2], axis = 0)
	

def process_input_table(input_file_name, comb_table, z_table, verbose = False, cols_to_use = [], \
							output_cols = ["drug_row","drug_col","conc_r","conc_C","response","conc_r_unit","conc_c_unit","cell"], \
							id_cols = [], output_name = "", enable_comboscore = False):

	"""
	Open and process the input table into the format:

	block_id,drug_row,drug_col,conc_r,conc_C,response,conc_r_unit,conc_c_unit,cell
	1,Everolimus,Dactolisib,0,0,90,nM,nM,BT-549
	1,Everolimus,Dactolisib,0,0.3,90,nM,nM,BT-549
	#SCORE
	#311 604 combinations
	"""
	grouped_table = comb_table.groupby(id_cols)
	output_dataframe = pd.DataFrame()
	block_id_count = 1
	failed_count = 0
	for entry in grouped_table:
		if verbose == True:
			print("Currently writing combination:",block_id_count,"/",len(grouped_table))
		try:
			current_table = entry[1]
			subset_table = current_table[cols_to_use]
			corrected_table = correct_units(subset_table)
			corrected_table.columns = output_cols

			#Add the rows with concentration 0
			zeros_sub_table = fetch_zeros(z_table, corrected_table, entry[0][1], entry[0][2], entry[0][0], \
											final_cols = output_cols, enable_comboscore = enable_comboscore)


			final_corrected_table = pd.concat([corrected_table, zeros_sub_table], axis = 0)
			final_corrected_table.insert(0, "block_id", [block_id_count]*final_corrected_table.shape[0])
			output_dataframe = pd.concat([output_dataframe, final_corrected_table], axis = 0)
			output_name = "tables/" + "_".join([str(x).replace("/","&") for x in entry[0]])
		except:
			print("Failed combination:", block_id_count)
			failed_count += 1
		block_id_count += 1
	output_dataframe.to_csv(DATASET_FOLDER + SYSTEM_SEP + output_prefix + INTERMEDIATE_SEP + input_file_name.split(SYSTEM_SEP)[-1], index = False)
	print("Failed:", failed_count, "combinations.")

usable_cols = ["NSC1","NSC2","CONC1","CONC2","PERCENTGROWTH","CONCUNIT1","CONCUNIT2","CELLNAME","SCORE"]
combo_table, zero_table = locate_concentration_zero(DATASETS_DICTIONARY["NCI_ALMANAC"], nan_col = "NSC2", cols_to_use = usable_cols, \
							left_columns = ["NSC1","CONC1","PERCENTGROWTH","CONCUNIT1","CELLNAME","SCORE"], \
							enable_comboscore = True, \
							output_cols_zeros = ["drug_name","concentration","response","unit","cell","combo_score"])
process_input_table(DATASETS_DICTIONARY["NCI_ALMANAC"], combo_table, zero_table, verbose = True, \
						cols_to_use = usable_cols, \
						id_cols = ["CELLNAME","NSC1","NSC2"], \
						output_cols = ["drug_row","drug_col","conc_r","conc_C","response","conc_r_unit","conc_c_unit","cell","combo_score"], \
						output_name = DATASETS_DICTIONARY["NCI_ALMANAC_comboscore"], enable_comboscore = True)