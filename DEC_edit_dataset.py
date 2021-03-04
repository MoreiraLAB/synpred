#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Edit the dataset to include full-agreement column and normalized (by mean and standard deviation) drug features 
conda activate tf
tensorflow version 1.15
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
from DEC_variables import SYSTEM_SEP, CSV_SEP, PARAGRAPH_SEP, \
							TAB_SEP, TRAIN_DATASET, TEST_DATASET, \
							TRAIN_DATASET_PROCESSED, TEST_DATASET_PROCESSED, \
							INTERMEDIATE_SEP, MORDRED_RAW_FILE, COLUMN_CLASSES, \
							COLUMN_CLASSES, FULL_AGREEMENT_COLUMN_NAME,\
							REDEPLOYMENT_FOLDER, SCALER_MORDRED_FILE, \
							CLASS_TRAIN_DATASET, CLASS_TEST_DATASET
from sklearn.preprocessing import StandardScaler
import pickle

def edit_file(input_file, output_name, class_table, mordred_dataset_file = MORDRED_RAW_FILE, \
				class_columns = COLUMN_CLASSES, mode = "train"):

	"""
	Open the file with the classes and mordred features, replace the features with 
	the correct scaling and add the full agreement class 
	"""
	opened_file = pd.read_csv(input_file, header = 0)
	#opened_class_file = pd.read_csv(class_table, header = 0)
	#print(opened_file.shape,opened_class_file.shape)
	#drop_classes = COLUMN_CLASSES + ["Loewe"]
	#opened_file = opened_file.drop(drop_classes, axis = 1)
	#opened_file = pd.concat([opened_file, opened_class_file], axis = 1)
	#print(opened_file)
	#sys.exit()
	#print(opened_file.shape)
	keep_columns = ["cell","drug1","drug2"] + COLUMN_CLASSES
	only_ids_and_class = opened_file[keep_columns]
	full_agreement_column = only_ids_and_class[COLUMN_CLASSES].apply(np.prod, axis = 1)
	only_ids_and_class[FULL_AGREEMENT_COLUMN_NAME] = full_agreement_column
	opened_mordred_file = pd.read_csv(mordred_dataset_file, header = 0)
	drug1_dataframe = opened_mordred_file.add_prefix("drug1_")
	drug1_dataframe["drug1"] = drug1_dataframe["drug1_NCI"]
	drug1_dataframe = drug1_dataframe.drop(["drug1_NCI"], axis = 1)
	joint_dataframe = only_ids_and_class.merge(drug1_dataframe, on = "drug1", how = "left")

	drug2_dataframe = opened_mordred_file.add_prefix("drug2_")
	drug2_dataframe["drug2"] = drug2_dataframe["drug2_NCI"]
	drug2_dataframe = drug2_dataframe.drop(["drug2_NCI"], axis = 1)
	
	final_dataframe = joint_dataframe.merge(drug2_dataframe, on = "drug2", how = "left")
	new_keep_columns = keep_columns + [FULL_AGREEMENT_COLUMN_NAME]
	final_ids = final_dataframe[new_keep_columns]
	only_features = final_dataframe.drop(new_keep_columns, axis = 1)
	features_name = list(only_features)
	scaler_name = REDEPLOYMENT_FOLDER + SYSTEM_SEP + SCALER_MORDRED_FILE
	if mode == "train":
		scaler = StandardScaler()
		scaler.fit(only_features)
		final_features = pd.DataFrame(scaler.transform(only_features))
		final_features.columns = features_name
		output_dataframe = pd.concat([final_ids, final_features], axis = 1)
		output_dataframe.to_csv(output_name, index = False)
		pickle.dump(scaler, open(scaler_name, 'wb'))
	elif mode == "other":
		scaler = pickle.load(open(scaler_name, 'rb'))
		final_features = pd.DataFrame(scaler.transform(only_features))
		final_features.columns = features_name
		output_dataframe = pd.concat([final_ids, final_features], axis = 1)
		output_dataframe.to_csv(output_name, index = False)

edit_file(TRAIN_DATASET, TRAIN_DATASET_PROCESSED, CLASS_TRAIN_DATASET, mode = "train")
edit_file(TEST_DATASET, TEST_DATASET_PROCESSED, CLASS_TEST_DATASET, mode = "other")