#!/usr/bin/env python

"""
Extract mordred features
"""

__author__ = "P. Matos-Filipe & A.J.Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SYNPRED"

import os
import subprocess
import sys
from rdkit import Chem
import pandas as pd
from mordred import Calculator, descriptors
import pickle
from sklearn.preprocessing import *
from synpred_variables import RESOURCES_FOLDER, SYSTEM_SEP

def mordredCalc(smilesString):
	
	"""
	Calculates descriptors from a SMILES string as input.
	Imput should be made as a list of SMILES strings.
	Outputs a list of dictionaries. Each dictionary as all the discriptors.
	"""
	mol = Chem.MolFromSmiles(smilesString)
	calc = Calculator(descriptors, ignore_3D = True)
	results = calc(mol)
	header = dict(results).keys()
	results = results.values()

	final = pd.Series(data=list(results), index = map(str, list(header)))
	return final

def drug_features_extractor(molecule, drug_index = "B", mode = "deploy"):

	"""
	Filters all features that returned stddev 0 during the making of the training dataset.
	Outputs data as a simple list of scaled features (using sklearn scaler from construction of the training dataset).
	Change the mode depending on whether you are fetching the features prior or after training.
	"""
	head = pickle.load(open(RESOURCES_FOLDER + SYSTEM_SEP + 'drug_features.pkl', 'rb'))
	finger_row = list(mordredCalc(molecule)[head])
	if mode == "fetch_only":
		return finger_row
	feature_scaler = pickle.load(open(RESOURCES_FOLDER + SYSTEM_SEP + 'scaler_mordred.pkl', 'rb'))
	if drug_index == "A":
		mean_values = feature_scaler.mean_[0:int(feature_scaler.mean_.shape[0] / 2)]
		var_values = feature_scaler.var_[0:int(feature_scaler.mean_.shape[0] / 2)]
	elif drug_index == "B":
		mean_values = feature_scaler.mean_[int(feature_scaler.mean_.shape[0] / 2):]
		var_values = feature_scaler.var_[0:int(feature_scaler.mean_.shape[0] / 2)]

	scaled_row = (finger_row - mean_values) / var_values
	return list(scaled_row)


##Example usage:
#sm_A = drug_features_extractor('CC(=O)OC1=CC=CC=C1C(=O)O')
#print(len(sm_A))