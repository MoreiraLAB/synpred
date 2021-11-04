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

from synpred_variables import SYSTEM_SEP, INTERMEDIATE_SEP, \
								DEFAULT_LOCATION, BLOCK_ID_COL, \
								DATASETS_DICTIONARY 

def split_file(input_file, blocks_by_split = 300, \
				target_folder = DEFAULT_LOCATION, group_col = BLOCK_ID_COL):

	"""
	Open the original file and split into several files
	"""
	import pandas as pd
	opened_file = pd.read_csv(input_file, sep = ",", header = 0)
	grouped_table = opened_file.groupby([group_col])
	output_group = []
	group_count, n = 1, 0
	for current_block in grouped_table:
		if n < blocks_by_split:
			output_group.append(current_block[1])
			n += 1
		if n == blocks_by_split:
			output_table = pd.concat(output_group, axis = 0)
			output_table_name = target_folder + SYSTEM_SEP + str(group_count) + INTERMEDIATE_SEP + input_file
			output_table.to_csv(output_table_name, index = False)
			output_group = []
			n = 0
			group_count += 1

split_file(DATASETS_DICTIONARY["NCI_ALMANAC_comboscore"], blocks_by_split = 1000)