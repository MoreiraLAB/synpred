#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process the DrugComb dataset to yield Full-agreement
"""

import pandas as pd
import sys
import synpred_variables as var

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

def exclude_non_pairs(input_table, target_columns = ["drug_row","drug_col"]):

    """
    Remove rows that do not have valid drug pairs
    """
    subset_table = input_table
    for current_column in target_columns:
        subset_table = subset_table[subset_table[current_column].notna()]
    return subset_table

def generate_full_agreement(input_table, usable_cols = ["synergy_zip","synergy_loewe","synergy_hsa","synergy_bliss","css_ri"]):

    """
    From the five available DrugComb synergy models ZIP, Loewe, HSA, Bliss and CSS, generate full agreement
    """
    def evaluate_synergy(input_list):

        """
        Full-agreement will only consider the rows in which all the synergy models are in agreement
        """
        synergy_value, synergy_value_val = 0, 0
        values_sum = sum(float(i) > 0 for i in input_list)
        if values_sum == 5:
            synergy_value_val = 1
            synergy_value = 1
        elif values_sum == 0:
            synergy_value_val = 1
            synergy_value = 0
        elif values_sum not in [0, 5]:
            synergy_value_val = 0
            synergy_value = 0
        return synergy_value, synergy_value_val

    full_agreement_col, full_agreement_val_col = [], []
    for index, row in input_table.iterrows():
        values_list = list(row[usable_cols].values)
        try:
            syn, syn_val = evaluate_synergy(values_list)
        except:
            syn, syn_val = 0, 0
        full_agreement_col.append(syn)
        full_agreement_val_col.append(syn_val)

    input_table["full_agreement"] = full_agreement_col
    input_table["full_agreement_val"] = full_agreement_val_col
    return input_table

def subset_cell_lines(input_table, cell_line_file = var.RESOURCES_FOLDER + var.SYSTEM_SEP + "almanac_cell_lines_library.csv", \
                        usable_cols = ["GDSC","NCI-60","CCLE"], \
                        target_col = "cell_line_name"):
    """
    Identify unique cell lines in the input file
    """

    opened_indexes = pd.read_csv(cell_line_file, sep = ",", header = 0, usecols = usable_cols)
    unique_list = []
    for current_column in usable_cols:
        unique_list += list(opened_indexes[current_column])
    output_table = input_table[input_table[target_col].isin(unique_list)]
    return output_table

def retrieve_smiles(input_table, drug_columns = ["drug_row","drug_col"], \
                    output_name = var.DATASETS_FOLDER + var.SYSTEM_SEP + "drugs_smiles.csv", verbose = True):

    """
    From the input table, identify the unique drugs and fetch the corresponding smile
    """
    from urllib.request import urlopen
    from urllib.parse import quote

    def CIRconvert(ids):
        #From thread https://stackoverflow.com/questions/54930121/converting-molecule-name-to-smiles
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans

    def chembl_fetch(input_id):

        """
        Access ChEMBL to retrieve canonical SMILES
        """
        from bioservices import ChEMBL
        chembl = ChEMBL()
        current_molecule = chembl.search_molecule(input_id)["molecules"][0]["molecule_structures"]["canonical_smiles"]
        return current_molecule

    def fetch_drugs(input_list, drugs_dictionary = {}):

        failed_drugs = []

        for i, current_drug in enumerate(input_list):
            try:
                smile = chembl_fetch(current_drug)
                if verbose == True:
                    print("Successfully got",current_drug, "\n", i + 1, "/", len(input_list))
            except:
                try:
                    smile = CIRconvert(current_drug)
                    if verbose == True:
                        print("Successfully got",current_drug, "\n", i + 1, "/", len(input_list))
                except:
                    if verbose == True:
                        print("Failed to get",current_drug, "\n", i + 1, "/", len(input_list))
                    failed_drugs.append(current_drug)
                    continue
            drugs_dictionary[current_drug] = [smile]
        return drugs_dictionary, failed_drugs

    drugs_list = []
    for current_column in drug_columns:
        drugs_list += list(input_table[current_column].unique())
    
    success_drugs_dict, failed_drugs_list = fetch_drugs(drugs_list)
    for i in range(5):
        success_drugs_dict, failed_drugs_list = fetch_drugs(failed_drugs_list, drugs_dictionary = success_drugs_dict)
    
    drugs_dataframe = pd.DataFrame.from_dict(success_drugs_dict).transpose()
    drugs_dataframe["drug_name"] = drugs_dataframe.index
    drugs_dataframe.columns = ["smile","drug_name"]
    drugs_dataframe.to_csv(output_name, sep = ";", index = False)    

def filter_for_existing_drugs(input_table, drugs_table, \
            drug_1 = "drug_row", drug_2 = "drug_col", smile_col = "smile", \
            output_name = var.DATASETS_FOLDER + var.SYSTEM_SEP + "combodb_for_synpred.csv"):
    """
    Retrieve the SMILES for the input entries
    """

    drug_1_table = drugs_table
    drug_1_table.columns = ["drug_1_smile","drug_1_name"]
    merged_drug_1 = input_table.merge(drug_1_table, left_on = drug_1, right_on = "drug_1_name")

    drug_2_table = drugs_table
    drug_2_table.columns = ["drug_2_smile","drug_2_name"]
    merged_drug_2 = merged_drug_1.merge(drug_2_table, left_on = drug_2, right_on = "drug_2_name")
    merged_drug_2.to_csv(output_name, sep = ";", index = False)


opened_drugcomb = pd.read_csv(var.DATASETS_DICTIONARY["DrugComb"], sep = var.CSV_SEP, header = 0)
opened_drugcomb = opened_drugcomb.loc[opened_drugcomb["study_name"] == "ALMANAC"]

subset_drugcomb = opened_drugcomb[["block_id","drug_row","drug_col","cell_line_name","tissue_name","synergy_zip","synergy_loewe","synergy_hsa","synergy_bliss","css_ri"]]
only_pairs_table = generate_full_agreement(exclude_non_pairs(subset_drugcomb))
only_pairs_table.to_csv(var.DATASETS_FOLDER + var.SYSTEM_SEP + "subset_drug_comb.csv", index = False)

only_pairs_table = pd.read_csv(var.DATASETS_FOLDER + var.SYSTEM_SEP + "subset_drug_comb.csv", sep = var.CSV_SEP, header = 0)
subset_cells_table = subset_cell_lines(only_pairs_table)
retrieve_smiles(subset_cells_table)

only_pairs_table = pd.read_csv(var.DATASETS_FOLDER + var.SYSTEM_SEP + "subset_drug_comb.csv", sep = var.CSV_SEP, header = 0)
subset_cells_table = subset_cell_lines(only_pairs_table)
drugs_table = pd.read_csv(var.DATASETS_FOLDER + var.SYSTEM_SEP + "drugs_smiles.csv", sep = ";", header = 0)
filter_for_existing_drugs(subset_cells_table, drugs_table)

opened_combodb_raw = pd.read_csv(var.DATASETS_DICTIONARY["combodb"], sep = ";", header = 0, usecols = ["cell_line_name", "drug_1_smile","drug_2_smile"])
opened_combodb_raw.columns = ["Cell", "Drug1", "Drug2"]
opened_combodb_raw.to_csv(var.DATASETS_DICTIONARY["final_drugcomb"], index = False, sep = ";")
