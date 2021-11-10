#!/usr/bin/env python

"""
Extract the omics based features
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SYNPRED"

import standalone_variables
from standalone_variables import INTERMEDIATE_SEP, SYSTEM_SEP, SEP, \
                                HOME, PROCESSED_TERMINATION, \
                                FEATURES_TERMINATION
from standalone_variables import CELL_LINES_FILE, CELL_LINES_ID, \
                                CELL_LINES_TYPE, CELL_LINES_COLUMN, \
                                CELL_LINES_5, CELL_LINES_6, \
                                CELL_LINES_7, CELL_LINES_12, CELL_LINES_14, \
                                CELL_LINES_17, CELL_LINES_22
import pandas as pd
import string
import sys
import os
from resources.feature_mining.mol_fingerprint import drug_features_extractor

def subset_table_rows(input_cell_line, input_file):

    """
    Identify the cells selected
    """
    opened_file = pd.read_csv(input_file, sep = SEP, header = 0)
    subset_table = opened_file.loc[opened_file[CELL_LINES_COLUMN].isin(input_cell_line)]
    return subset_table

def locate_common(input_tables = ["","",""], target_column = CELL_LINES_COLUMN):

    """
    Fetch the matching cell lines
    """
    all_lists = []
    for current_table in input_tables:
        all_lists.append(list(current_table[target_column]))
    first_comparison = list(set(all_lists[0]).intersection(all_lists[1]))
    final_list = list(set(first_comparison).intersection(all_lists[2]))
    return final_list

def empty_zeros(input_table, current_cell):

    """
    If the cell lines is not found, replace the values with 0s as was validated in the pipeline
    """
    if input_table.empty == True:
        return pd.DataFrame([[0]*(input_table.shape[1] - 1) + [current_cell]], columns = list(input_table))
    else:
        return input_table

def generate_features_file(input_cell_line, input_drug_A, input_drug_B):

    """
    Fetch the 7 blocks of CCLE features and calculate the mordred features
    Sorry for the redundancy, but it seemed a better way to keep track of the features blocks
    """
    cell_lines_list = standalone_variables.process_cells_file(CELL_LINES_FILE, \
                                    mode = "features_extraction",  \
                                    target_column = CELL_LINES_TYPE,  \
                                    input_cell_type = input_cell_line)
    cell_features_5 = empty_zeros(subset_table_rows(cell_lines_list, CELL_LINES_5).add_prefix("CCLE_5_"), input_cell_line)
    cell_features_5.rename(columns={'CCLE_5_cells':'cells'}, inplace=True)
    cell_features_6 = empty_zeros(subset_table_rows(cell_lines_list, CELL_LINES_6).add_prefix("CCLE_6_"), input_cell_line)
    cell_features_6.rename(columns={'CCLE_6_cells':'cells'}, inplace=True)
    cell_features_7 = empty_zeros(subset_table_rows(cell_lines_list, CELL_LINES_7).add_prefix("CCLE_7_"), input_cell_line)
    cell_features_7.rename(columns={'CCLE_7_cells':'cells'}, inplace=True)
    cell_features_12 = empty_zeros(subset_table_rows(cell_lines_list, CELL_LINES_12).add_prefix("CCLE_12_"), input_cell_line)
    cell_features_12.rename(columns={'CCLE_12_cells':'cells'}, inplace=True)
    cell_features_14 = empty_zeros(subset_table_rows(cell_lines_list, CELL_LINES_14).add_prefix("CCLE_14_"), input_cell_line)
    cell_features_14.rename(columns={'CCLE_14_cells':'cells'}, inplace=True)
    cell_features_17 = empty_zeros(subset_table_rows(cell_lines_list, CELL_LINES_17).add_prefix("CCLE_17_"), input_cell_line)
    cell_features_17.rename(columns={'CCLE_17_cells':'cells'}, inplace=True)
    cell_features_22 = empty_zeros(subset_table_rows(cell_lines_list, CELL_LINES_22).add_prefix("CCLE_22_"), input_cell_line)
    cell_features_22.rename(columns={'CCLE_22_cells':'cells'}, inplace=True)
    common_cell_lines = locate_common([cell_features_5, cell_features_6, cell_features_7, \
                                        cell_features_12, cell_features_14, cell_features_17, \
                                        cell_features_22])
    cell_5_filtered = cell_features_5.loc[cell_features_5[CELL_LINES_COLUMN].isin(common_cell_lines)]
    cell_6_filtered = cell_features_6.loc[cell_features_6[CELL_LINES_COLUMN].isin(common_cell_lines)]
    cell_7_filtered = cell_features_7.loc[cell_features_7[CELL_LINES_COLUMN].isin(common_cell_lines)]
    cell_12_filtered = cell_features_12.loc[cell_features_12[CELL_LINES_COLUMN].isin(common_cell_lines)]
    cell_14_filtered = cell_features_14.loc[cell_features_14[CELL_LINES_COLUMN].isin(common_cell_lines)]
    cell_17_filtered = cell_features_17.loc[cell_features_17[CELL_LINES_COLUMN].isin(common_cell_lines)]
    cell_22_filtered = cell_features_22.loc[cell_features_22[CELL_LINES_COLUMN].isin(common_cell_lines)]
    two_genomic_features = pd.merge(cell_5_filtered, cell_6_filtered, on = CELL_LINES_COLUMN, how = "left")
    three_genomic_features = pd.merge(two_genomic_features, cell_7_filtered, on = CELL_LINES_COLUMN, how = "left")
    four_genomic_features = pd.merge(three_genomic_features, cell_12_filtered, on = CELL_LINES_COLUMN, how = "left")
    five_genomic_features = pd.merge(four_genomic_features, cell_14_filtered, on = CELL_LINES_COLUMN, how = "left")
    six_genomic_features = pd.merge(five_genomic_features, cell_17_filtered, on = CELL_LINES_COLUMN, how = "left")
    final_genomic_features = pd.merge(six_genomic_features, cell_22_filtered, on = CELL_LINES_COLUMN, how = "left")
    drug_A_features = pd.DataFrame(drug_features_extractor(input_drug_A)).transpose()
    drug_B_features = pd.DataFrame(drug_features_extractor(input_drug_B)).transpose()

    drug_features = pd.concat([pd.concat([drug_A_features, drug_B_features], axis = 1)]*final_genomic_features.shape[0], axis = 0).reset_index()
    all_features = pd.concat([final_genomic_features, drug_features], axis = 1).drop(["index","cells"],axis = 1)
    return all_features