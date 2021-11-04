#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge the table after calculation with synergyfinder
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

import pandas as pd
import os
import sys

from synpred_variables import DEFAULT_LOCATION, SYSTEM_SEP, \
								INTERMEDIATE_SEP, CSV_SEP, SPLIT_TABLES_FOLDER, \
								BLOCK_ID_COL, NCI_ALMANAC_ALL_CLASSES, \
								DATASETS_DICTIONARY, \
								COMBOSCORE_NCI_ALMANAC, ONLY_COMBINATIONS_NCI_ALMANAC

def detect_unique_concentrations(input_sub_table, input_raw_sub_table):

	"""
	Compare the concentrations on the table before and after synergyfinder,
	as it might have swapped drug1 with drug2
	"""
	unique_concentrations_1 = sorted([float(x) for x in list(input_sub_table["conc1"].unique())])
	unique_concentrations_2 = sorted([float(x) for x in list(input_sub_table["conc2"].unique())])
	unique_raw_concentrations_1 = sorted([float(x) for x in list(input_raw_sub_table["Conc1"].unique())])
	unique_raw_concentrations_2 = sorted([float(x) for x in list(input_raw_sub_table["Conc2"].unique())])
	if (unique_concentrations_1 == unique_raw_concentrations_1) and (unique_concentrations_2 == unique_raw_concentrations_2):
		return False
	else:
		return True

def confirm_table(input_sub_table, input_raw_sub_table, drug_swap_value = False):

	"""
	Merge the table based on concentration and drug columns
	"""
	if drug_swap_value == True:
		renamed_raw_table = input_raw_sub_table.rename(columns = {'Drug1': 'Drug2', 'Drug2': 'Drug1',"Conc1":"Conc2","Conc2":"Conc1"}, inplace = False)
	elif drug_swap_value == False:
		renamed_raw_table = input_raw_sub_table
	rearranjed_df = pd.merge(input_sub_table, renamed_raw_table,  how = 'left', left_on = ['conc1','conc2'], right_on = ['Conc1','Conc2'])
	rearranjed_df["response"].loc[(rearranjed_df["conc1"] == 0.0) & (rearranjed_df["conc2"] == 0.0)] = 100
	rearranjed_df["combo_score"].loc[(rearranjed_df["conc1"] == 0.0) & (rearranjed_df["conc2"] == 0.0)] = 0
	rearranjed_df["Drug1"].loc[(rearranjed_df["conc1"] == 0.0) & (rearranjed_df["conc2"] == 0.0)] = [x for x in rearranjed_df["Drug1"].unique() if x == x][0]
	rearranjed_df["Drug2"].loc[(rearranjed_df["conc1"] == 0.0) & (rearranjed_df["conc2"] == 0.0)] = [x for x in rearranjed_df["Drug2"].unique() if x == x][0]
	rearranjed_df["cell"].loc[(rearranjed_df["conc1"] == 0.0) & (rearranjed_df["conc2"] == 0.0)] = [x for x in rearranjed_df["cell"].unique() if x == x][0]
	rearranjed_df = rearranjed_df.dropna(axis = 0)
	return rearranjed_df[["block_id_x","conc1","conc2","ZIP",\
							"HSA","Bliss","Loewe",\
							"Drug1","Drug2","response","combo_score","cell"]].rename(columns = {'block_id_x': 'block_id'})

def merge_class_tables(input_folder, base_file, \
						file_init = "calculated_class", output_file = "NCI_ALMANAC_drug_combinations.csv", \
						usable_cols = ["block_id","conc1","conc2","ZIP","Loewe","HSA","Bliss"], \
						usable_raw_cols = ["block_id","Drug1","Drug2","Conc1","Conc2","response","combo_score","cell"], \
						id_col = "block_id", verbose = False, mode = ""):

	"""
	Iterate over the calculated class after running synergyfinder,
	combine it with the initial "ComboScore" and yield an output file 
	"""
	folder_loc = DEFAULT_FOLDER + SYSTEM_SEP + input_folder
	opened_raw_file = pd.read_csv(base_file, sep = CSV_SEP, header = 0, usecols = usable_raw_cols)
	grouped_raw_ids = dict(tuple(opened_raw_file.groupby(["block_id"])))
	output_list = []
	for files in os.listdir(folder_loc):
		if files.startswith(file_init):
			if verbose == True:
				print("Currently evaluating:", files)
			file_loc = folder_loc + SYSTEM_SEP + files
			opened_table = pd.read_csv(file_loc, sep = CSV_SEP, header = 0, usecols = usable_cols)
			
			if mode == "single_combo":
				sub_grouped_ids = opened_table.groupby(["block_id"])
				for current_id_group in sub_grouped_ids:
					"""
					Groupby yields iterable tuples in which the first element is 
					the grouped value and the second the table subset for that value 
					"""
					output_table = current_id_group[1]
					output_table["cell"] = grouped_raw_ids[current_id_group[0]]["cell"].unique()[0]
					output_list.append(output_table.rename(columns = {"drug1":"Drug1","drug2":"Drug2"}))
				continue
			sub_grouped_ids = opened_table.groupby(["block_id"])
			for current_id_group in sub_grouped_ids:
				"""
				Groupby yields iterable tuples in which the first element is 
				the grouped value and the second the table subset for that value 
				"""
				swapped_tag = detect_unique_concentrations(current_id_group[1], grouped_raw_ids[current_id_group[0]])
				arranged_table = confirm_table(current_id_group[1], grouped_raw_ids[current_id_group[0]], drug_swap_value = False)
				output_list.append(arranged_table)
	merged_total_table = pd.concat(output_list, axis = 0)
	merged_total_table.to_csv(output_file, index = False, sep = CSV_SEP)

merge_class_tables(SPLIT_TABLES_FOLDER, DATASETS_DICTIONARY["NCI_ALMANAC_comboscore"], \
					verbose = True, file_init = "combo_calculated_class", \
					output_file = DATASETS_DICTIONARY["NCI_ALMANAC_classes"]), \
merge_class_tables(SPLIT_TABLES_FOLDER, DATASETS_DICTIONARY["NCI_ALMANAC_comboscore"], \
					verbose = True, file_init = "pairs_calculated_class", \
					output_file = DATASETS_DICTIONARY["NCI_ALMANAC_only_combinations"], \
					mode = "single_combo", usable_cols = ["block_id","ZIP","Loewe","HSA","Bliss"])