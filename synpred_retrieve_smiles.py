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
								INTERMEDIATE_SEP
import sys
import pubchempy

def identify_unique_drugs(input_table, drug_1_col = "Drug1", drug_2_col = "Drug2"):

	"""
	Identify the unique drugs and download smile
	"""
	unique_drugs_1 = list(input_table[drug_1_col].unique())
	unique_drugs_2 = list(input_table[drug_2_col].unique())
	return list(set(unique_drugs_1 + unique_drugs_2))

def download_unique_smiles(input_drugs_list, verbose = True):

	"""
	Download unique smiles
	"""
	from pubchempy import get_compounds
	for current_drug in input_drugs_list:
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


opened_table = pd.read_csv(RAW_EXAMPLE_TABLE, sep = CSV_SEP)
unique_drugs = identify_unique_drugs(opened_table)
download_unique_smiles(unique_drugs)