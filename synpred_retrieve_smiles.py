#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Retrieve the smiles from the input table
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

import pandas as pd
from synpred_variables import SYSTEM_SEP, CSV_SEP, MOLECULES_FOLDER, \
								RAW_EXAMPLE_TABLE, TXT_TERMINATION, \
								INTERMEDIATE_SEP, DATASETS_DICTIONARY
import sys
from synpred_support_functions import identify_unique_drugs

def download_unique_smiles(input_drugs_list, verbose = True, mode = "compound_name"):

	"""
	Download unique smiles
	"""
	import os.path
	from os import path
	if mode == "compound_name":
		from pubchempy import get_compounds
	elif mode == "NSC":
		from synpred_support_functions import extract_smile_from_NSC
	count = 1
	for current_drug in input_drugs_list:
		if mode == "compound_name":
			try:
				current_smile = get_compounds(current_drug, 'name')[0].isomeric_smiles
				output_name = MOLECULES_FOLDER + SYSTEM_SEP + current_drug.replace(" ",INTERMEDIATE_SEP) + TXT_TERMINATION
				with open(output_name, "w") as smile_file:
					smile_file.write(current_smile)
				if verbose == True:
					print("Successfully downloaded:", current_drug)
			except:
				if verbose == True:
					print("Failed to download:", current_drug)
				else:
					continue
		elif mode == "NSC":
			if verbose == True:
				print("Evaluating drug NSC:", str(int(current_drug)), "entry:", count , "/", str(len(input_drugs_list)))
			output_name = MOLECULES_FOLDER + SYSTEM_SEP + str(int(current_drug)) + TXT_TERMINATION
			if path.exists(output_name):
				print("--Drug already has been retrieved")
				count += 1
				continue
			else:
				print("--Retrieving:", str(int(current_drug)))
				current_smile = extract_smile_from_NSC(current_drug)
				with open(output_name, "w") as smile_file:
					smile_file.write(current_smile)
		count += 1

opened_table = pd.read_csv(DATASETS_DICTIONARY["NCI_ALMANAC"], sep = CSV_SEP, usecols = ["NSC1","NSC2"])
unique_drugs = [x for x in identify_unique_drugs(opened_table, drug_1_col = "NSC1", drug_2_col = "NSC2") if str(x) != "nan"]
download_unique_smiles(unique_drugs, mode = "NSC")