#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate molecular descriptor features
conda activate tf
tensorflow version 1.15
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"


from DEC_variables import MOLECULES_FOLDER, SYSTEM_SEP, \
                            CSV_SEP, SMILE_TERMINATION, \
                            SDF_TERMINATION, PARAGRAPH_SEP, \
                            INTERMEDIATE_SEP, CSV_TERMINATION, \
                            CUSTOM_MOLECULES_FEATURES
import os
import pandas as pd
import sys
import numpy as np

def generate_raw_tables(input_folder = MOLECULES_FOLDER, verbose = True):

    """
    Generate coordinates, bonds and weight tables
    """
    for file_name in os.listdir(input_folder):
        if verbose == True:
            print("Currently evaluating:", file_name.split(".")[0])
        if file_name.endswith(SDF_TERMINATION):
            file_location = MOLECULES_FOLDER + SYSTEM_SEP + file_name
            coordinates_location = MOLECULES_FOLDER + SYSTEM_SEP + \
                    file_name.split(".")[0] + INTERMEDIATE_SEP + "coordinates" + \
                    CSV_TERMINATION
            bonds_location = MOLECULES_FOLDER + SYSTEM_SEP + \
                    file_name.split(".")[0] + INTERMEDIATE_SEP + "bonds" + \
                    CSV_TERMINATION
            weight_location = MOLECULES_FOLDER + SYSTEM_SEP + \
                    file_name.split(".")[0] + INTERMEDIATE_SEP + "weight" + \
                    CSV_TERMINATION
            coordinates_table = []
            bonds_table = []
            opened_file = open(file_location, "r").readlines()
            detected_weight = False
            for row in opened_file:
                current_row = row.replace(PARAGRAPH_SEP, "").split()
                row_length = len(current_row)
                if row_length == 16:
                    coordinates_table.append([current_row[0], current_row[1], current_row[3]])
                if row_length == 7:
                    bonds_table.append(current_row)
                if detected_weight == False:
                    if row.replace(" ","").replace(PARAGRAPH_SEP,"") == "><MolecularWeight(nearestinteger)>":
                        detected_weight = True
                elif detected_weight == True:
                    weight = row
                    break
            coordinates_dataframe = pd.DataFrame(coordinates_table, columns = ["x","y","atom_type"])
            bonds_dataframe = pd.DataFrame(bonds_table, columns = ["bond_atom_1","bond_atom_2","bond_type_1","bond_type_2",\
                                                                "bond_type_3","bond_type_4","bond_type_5"])
            opened_weight = open(weight_location,"w").write(weight + PARAGRAPH_SEP)
            coordinates_dataframe.to_csv(coordinates_location, index = False)
            bonds_dataframe.to_csv(bonds_location, index = False)

def detect_dimensions(input_folder = MOLECULES_FOLDER, mode = "analyse_content"):

    """
    Detect the maximum amount of atoms per molecule
    maximum coordinates = 110
    {'C': 2030, 'O': 516, 'N': 327, 'P': 7, 'H': 76, 
    'Cl': 51, 'Na': 1, 'F': 31, 'S': 26, 'Br': 3,
     'As': 2, 'B': 1, 'Pt': 3}
    atom types = 13

    Generate empty dataframe row
    """
    maximum_coordinates = 0
    unique_type_list = []
    count_dictionary = {}
    for file_name in os.listdir(input_folder):
        if file_name.endswith("coordinates" + CSV_TERMINATION):
            file_location = MOLECULES_FOLDER + SYSTEM_SEP + file_name
            coordinates_table = pd.read_csv(file_location, sep = CSV_SEP, header = 0)
            if coordinates_table.shape[0] > maximum_coordinates:
                maximum_coordinates = coordinates_table.shape[0]
            unique_atoms = list(coordinates_table["atom_type"].unique())
            counts_table = coordinates_table["atom_type"].value_counts()
            unique_type_list = list(set(unique_type_list + unique_atoms))
    column_names = ["drug"]
    for coordinates_name in range(1,maximum_coordinates + 1):
        column_names += ["x" + INTERMEDIATE_SEP + str(coordinates_name), \
                        "y" + INTERMEDIATE_SEP + str(coordinates_name)]
        atom_type_list = []
        for atom_name in unique_type_list:
            atom_type_list.append(atom_name + INTERMEDIATE_SEP + str(coordinates_name))
        column_names += atom_type_list
        bond_type_list =[]
        for bond_type in ["bond_type_1","bond_type_2",\
                        "bond_type_3","bond_type_4","bond_type_5"]:
            bond_type_list.append(bond_type + INTERMEDIATE_SEP + str(coordinates_name))
        column_names += bond_type_list
    column_names.append("molecule_weight")
    return pd.DataFrame(np.zeros((1,len(column_names))), columns = column_names)

def generate_features_table(input_folder = MOLECULES_FOLDER, \
                                output_file = CUSTOM_MOLECULES_FEATURES, \
                                verbose = True):

    """
    Generate a table with all the output features.
    Firstly, generate a open row

    """
    #pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 100)
    #pd.set_option('display.width', 1000)
    #local empty_row
    #empty_row = detect_dimensions()
    started_writing = False
    output_dataframe = pd.DataFrame(columns = list(detect_dimensions()))
    for file_name in os.listdir(input_folder):
        if file_name.endswith("coordinates" + CSV_TERMINATION):
            current_features_row = detect_dimensions()
            molecule_id = file_name.split(INTERMEDIATE_SEP)[0]
            if verbose == True:
                print("Currently merging:", molecule_id)
            coordinates_file_location = MOLECULES_FOLDER + SYSTEM_SEP + molecule_id + \
                                        INTERMEDIATE_SEP + "coordinates" + CSV_TERMINATION
            bonds_file_location = MOLECULES_FOLDER + SYSTEM_SEP + molecule_id + \
                                        INTERMEDIATE_SEP + "bonds" + CSV_TERMINATION
            weight_file_location = MOLECULES_FOLDER + SYSTEM_SEP + molecule_id + \
                                        INTERMEDIATE_SEP + "weight" + CSV_TERMINATION
            opened_coordinates = pd.read_csv(coordinates_file_location, header = 0)

            for index, row in opened_coordinates.iterrows():
                x_col_name = "x" + INTERMEDIATE_SEP + str(index + 1)
                y_col_name = "y" + INTERMEDIATE_SEP + str(index + 1)
                atom_type_name = row["atom_type"] + INTERMEDIATE_SEP + str(index + 1)
                current_features_row[x_col_name] = row["x"]
                current_features_row[y_col_name] = row["y"]
                current_features_row[atom_type_name] = 1

            current_features_row["drug"] = molecule_id
            opened_bonds = pd.read_csv(bonds_file_location, header = 0)
            for index_bond, row_bond in opened_bonds.iterrows():
                first_atom = row_bond["bond_atom_1"]
                second_atom = row_bond["bond_atom_2"]
                for bond_type in ["bond_type_1","bond_type_2",\
                        "bond_type_3","bond_type_4","bond_type_5"]:
                    current_bond_type_first = bond_type + INTERMEDIATE_SEP + str(first_atom)
                    current_bond_type_second = bond_type + INTERMEDIATE_SEP + str(second_atom)
                    try:
                        current_features_row[current_bond_type_first] += row_bond[bond_type]
                        current_features_row[current_bond_type_second] += row_bond[bond_type]
                    except:
                        continue
            current_features_row["molecule_weight"] =  open(weight_file_location, "r").readlines()[0].replace(PARAGRAPH_SEP,"")
            output_dataframe = pd.concat([output_dataframe, current_features_row], axis = 0)
    output_dataframe.to_csv(output_file, index = False)

"""
Firstly, generate the individual features tables
generate_raw_tables()
Then, generate the full table
"""
#generate_raw_tables()
generate_features_table()