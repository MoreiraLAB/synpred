#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Join the datasets on their final stage
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

from synpred_variables import CSV_SEP, SYSTEM_SEP, PARAGRAPH_SEP, \
                            INTERMEDIATE_SEP, TAB_SEP, \
                            SCALED_CCLE_START, CCLE_FOLDER_PROCESSED

from synpred_variables import DEFAULT_LOCATION, CCLE_FOLDER, \
                            SUPPORT_FOLDER, CCLE_DATASET_LOG_FILE, \
                            REDEPLOYMENT_FOLDER, CCLE_ID_COLUMN_NAME, \
                            CCLE_ID_COLUMN_SEP, SUBSET_CCLE, FILTERED_CCLE, \
                            CCLE_ANNOTATION_FILE, NCI_ALMANAC_CELL_COLUMN, \
                            DATASETS_DICTIONARY, PCA_CCLE, \
                            CCLE_COLUMN_TAG, AUTOENCODER_CCLE, \
                            AUTOENCODER_LOG_FILE, RANDOM_STATE, \
                            MORDRED_RAW_FILE, SCALER_MORDRED_FILE, \
                            MORDRED_PROCESSED_FILE

from synpred_support_functions import open_log_file, alternative_ID_file
import os
import pandas as pd
import tensorflow as tf
import sys
import numpy as np
import pickle
import random

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def get_template_tables(logger_file, alternative_ids_dictionary):

    """
    Get the template tables with a new column adding the matching ID
    """
    output_dictionary = {}
    for current_subset in logger_file.keys():
        subset_file_name = CCLE_FOLDER + SYSTEM_SEP + current_subset + INTERMEDIATE_SEP + FILTERED_CCLE
        opened_file = pd.read_csv(subset_file_name)
        alternative_ids = logger_file[current_subset]
        new_index_column = []
        for index, row in opened_file.iterrows():
            current_class = row[0]
            split_class_column = row[0].split(CCLE_ID_COLUMN_SEP)
            if len(split_class_column) > 1:
                for class_name in alternative_ids_dictionary.keys():
                    for optional_class in split_class_column:
                        if optional_class in alternative_ids_dictionary[class_name]:
                            new_index_column.append(optional_class)
                            break
            elif len(split_class_column) == 1:
                for class_name in alternative_ids_dictionary.keys():
                    if current_class in alternative_ids_dictionary[class_name]:
                        new_index_column.append(class_name)
                        break
        opened_file["cell"] = new_index_column
        output_dictionary[current_subset] = opened_file
    return output_dictionary

def reconsider_id_column(input_table, target_column = "cell", template_table = alternative_ID_file()):

    """
    Replace the column id with NCI-Almanac correspondent
    """
    new_column = []
    for current_entry in input_table[target_column]:
        detected = False
        for current_key in template_table.keys():
            if detected == False:
                for x in template_table[current_key]:
                    if current_entry == x:
                        new_column.append(current_key)
                        detected = True
                        break
        if detected == False:
            new_column.append(current_entry)
    input_table[target_column] = new_column
    return input_table

def autoencoder(input_table, input_size, input_architecture, subset_name, \
                write_mode = True, number_of_epochs = 1000, model_name = ""):

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LeakyReLU, \
                                    BatchNormalization
    import tensorflow as tf
    from tensorflow import keras
    """
    Perform dimensionality reduction with autoencoder
    """
    input_layer = Input(shape = (input_size,))
    # encoder level 1
    encoder = Dense(input_architecture[0])(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU()(encoder)

    # encoder level 2
    encoder = Dense(input_architecture[1])(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU()(encoder)

    # bottleneck
    bottleneck = Dense(input_architecture[2])(encoder)

    # define decoder, level 1
    decoder = Dense(input_architecture[3])(bottleneck)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)
    # decoder level 2
    decoder = Dense(input_architecture[4])(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)
    # output layer
    output = Dense(input_size, activation = 'linear')(decoder)
    # define autoencoder model
    model = Model(inputs = input_layer, outputs = output)
    # compile autoencoder model
    model.compile(optimizer = "adam", loss = 'mse')
    model.fit(input_table, input_table, epochs = number_of_epochs, verbose = True)

    dimensionality_model = Model(inputs = input_layer, outputs = bottleneck)
    intermediate_table = model.predict(input_table)
    output_table = dimensionality_model.predict(input_table)

    print("RMSE:",np.mean(np.sqrt(np.mean((intermediate_table-input_table)**2))))
    if write_mode == True:
        with open(AUTOENCODER_LOG_FILE, "a") as log_file:
            to_write_row = subset_name + CSV_SEP + \
                str(np.mean(np.sqrt(np.mean((intermediate_table-input_table)**2)))) + PARAGRAPH_SEP
            log_file.write(to_write_row)
            dimensionality_model.save(model_name, save_format = "h5")
    return output_table, dimensionality_model

def explain_PCA_variance(input_PCA, output_name, input_table, dim_number, output_var_name):

    """
    Get feature importance from PCA pre-processing
    """
    PCA_header = ["PC_" + str(i) for i in range(1, dim_number + 1)]
    components_contribution = pd.DataFrame(input_PCA.components_, columns = list(input_table))
    components_contribution["PC"] = PCA_header
    components_contribution.to_csv(output_name, index = False)
    var_df = pd.DataFrame(input_PCA.explained_variance_ratio_.reshape(1,len(PCA_header)), columns = PCA_header)
    var_df.to_csv(output_var_name, index = False)

class process_CCLE:

    def __init__(self, verbose = True, starting_folder = CCLE_FOLDER, \
                    target_folder = CCLE_FOLDER_PROCESSED, \
                    logging_file = CCLE_DATASET_LOG_FILE, \
                    cell_col = "cell"):

        self.verbose = verbose
        self.CCLE_subsets = ["5","6","7","12","14","17","22"]
        self.train_cell_lines = list(pd.read_csv(SUPPORT_FOLDER + SYSTEM_SEP + "train_cell_lines.csv", sep = CSV_SEP, header = None).iloc[:,0])
        self.test_cell_lines = list(pd.read_csv(SUPPORT_FOLDER + SYSTEM_SEP + "test_cell_lines.csv", sep = CSV_SEP, header = None).iloc[:,0])
        self.all_cell_lines = self.train_cell_lines + self.test_cell_lines
        self.starting_folder = starting_folder
        self.target_folder = target_folder
        self.logger = open_log_file(logging_file)
        self.alternative_dictionary = alternative_ID_file()
        self.tables_dictionary = get_template_tables(self.logger, self.alternative_dictionary)
        self.cell = cell_col

    def normalize_features(self, write_pkl = True):

        from sklearn.preprocessing import StandardScaler
        self.dataframes_normalized_dictionary = {}
        for current_subset in self.logger.keys():
            subset_table = self.tables_dictionary[current_subset].iloc[:,1:].add_prefix(CCLE_COLUMN_TAG + INTERMEDIATE_SEP + current_subset + INTERMEDIATE_SEP)
            subset_table = reconsider_id_column(subset_table.rename(columns = {CCLE_COLUMN_TAG + INTERMEDIATE_SEP + current_subset + INTERMEDIATE_SEP + self.cell: self.cell}))
            try:
                subset_table = subset_table.drop(["Unnamed: 0.1"], axis = 1)
            except:
                pass
            
            features_columns = list(subset_table.drop([self.cell], axis = 1))

            train_subset = subset_table.loc[subset_table[self.cell].isin(self.train_cell_lines)]
            test_subset = subset_table.loc[subset_table[self.cell].isin(self.test_cell_lines)]

            ID_train = list(train_subset[self.cell])
            ID_test = list(test_subset[self.cell])

            current_scaler = StandardScaler()
            current_scaler.fit(train_subset.drop([self.cell], axis = 1))
            if write_pkl == True:
                output_pkl_name = REDEPLOYMENT_FOLDER + SYSTEM_SEP + "scaler_CCLE_" + current_subset + ".pkl"
                with open(output_pkl_name,'wb') as output_scaler:
                    pickle.dump(current_scaler, output_scaler)
            
            transformed_train = pd.DataFrame(current_scaler.transform(train_subset.drop([self.cell], axis = 1)), columns = features_columns)
            transformed_train[self.cell] = ID_train

            transformed_test = pd.DataFrame(current_scaler.transform(test_subset.drop([self.cell], axis = 1)), columns = features_columns)
            transformed_test[self.cell] = ID_test
            self.dataframes_normalized_dictionary[current_subset] = {"train": transformed_train, "test": transformed_test}

    def reduce_CCLE(self, reduce_mode = "PCA", number_of_dimensions = 50, \
                            write_feature_importances = False, output_names_dictionary = {}):
        
        from sklearn.decomposition import PCA
        import pickle

        first_key = False
        for CCLE_subset in self.CCLE_subsets:
            output_name = CCLE_FOLDER_PROCESSED + SYSTEM_SEP + reduce_mode + \
                INTERMEDIATE_SEP + CCLE_subset + ".csv"

            train_data = self.dataframes_normalized_dictionary[CCLE_subset]["train"].drop([self.cell], axis = 1)
            train_ids = list(self.dataframes_normalized_dictionary[CCLE_subset]["train"][self.cell])
            test_data  = self.dataframes_normalized_dictionary[CCLE_subset]["test"].drop([self.cell], axis = 1)
            test_ids = list(self.dataframes_normalized_dictionary[CCLE_subset]["test"][self.cell])
            if reduce_mode == "PCA":
                pca = PCA(n_components = number_of_dimensions)
                pca.fit(train_data)
                if write_feature_importances == True:
                    output_importance_name = SUPPORT_FOLDER + SYSTEM_SEP  + \
                                CCLE_subset + INTERMEDIATE_SEP + "importance" + INTERMEDIATE_SEP + PCA_CCLE
                    output_variance_name = SUPPORT_FOLDER + SYSTEM_SEP + \
                                CCLE_subset + INTERMEDIATE_SEP + "variance" + INTERMEDIATE_SEP + PCA_CCLE
                    output_PCA_name = REDEPLOYMENT_FOLDER + SYSTEM_SEP + \
                                CCLE_subset + INTERMEDIATE_SEP + "PCA_transform.pkl"
                    with open(output_PCA_name,'wb') as PCA_file:
                        pickle.dump(pca, PCA_file)
                    if self.verbose == True:
                        print("Currently writing:", output_importance_name)
                    explain_PCA_variance(pca, output_importance_name, features_columns, number_of_dimensions, output_variance_name)
                reduced_train = pd.DataFrame(pca.transform(train_data)).add_prefix("CCLE_" + CCLE_subset + "_")
                reduced_train[self.cell] = train_ids

                reduced_test = pd.DataFrame(pca.transform(test_data)).add_prefix("CCLE_" + CCLE_subset + "_")
                reduced_test[self.cell] = test_ids

            if reduce_mode == "autoencoder":
                output_encoder_name = REDEPLOYMENT_FOLDER + SYSTEM_SEP + \
                                         INTERMEDIATE_SEP + CCLE_subset + INTERMEDIATE_SEP + "autoencoder.h5"
                features_size = train_data.shape[1]
                sample_size = train_data.shape[0]
                architecture = []
                if sample_size <= 100 or features_size <= 100:
                    epochs = 1000
                if ((sample_size > 100) and (sample_size < 1000)) or ((features_size > 100) and (features_size <= 250)):
                    epochs = 250
                if sample_size >= 1000 or features_size > 250:
                    epochs = 100
                if features_size <= 250:
                    architecture = [features_size * 2, features_size, int(features_size / 2), features_size, features_size * 2]
                if features_size > 250 and features_size <= 1000:
                    architecture = [features_size, int(features_size / 2), int(features_size / 10), int(features_size/2), features_size]
                if features_size > 1000:
                    architecture = [int(features_size / 2), int(features_size / 4), int(features_size / 50), int(features_size / 4), int(features_size / 2)]
                reduced_train, autoencoder_model = autoencoder(train_data, features_size, \
                                            architecture, CCLE_subset, number_of_epochs = epochs, \
                                            model_name = output_encoder_name)
    
                reduced_train = pd.DataFrame(reduced_train)
                reduced_train[self.cell] = train_ids

                reduced_test = pd.DataFrame(autoencoder_model.predict(test_data))
                reduced_test[self.cell] = test_ids
            
            if self.verbose == True:
                print("Successfully reduced", CCLE_subset, "with a", reduce_mode ,\
                    "to a shape of: \n --Train:", reduced_train.shape, \
                    "\n --Test:", reduced_train.shape)
            output_table = pd.concat([reduced_train, reduced_test], axis = 0)
            output_table.to_csv(output_name, sep = CSV_SEP, index = False)

def normalize_drugs(input_table_file = MORDRED_RAW_FILE, train_drugs_file = SUPPORT_FOLDER + SYSTEM_SEP + "train_drugs.csv", \
                        test_drugs_file = SUPPORT_FOLDER + SYSTEM_SEP + "test_drugs.csv", \
                        mordred_drug_ID_col = "NCI", \
                        output_file = MORDRED_PROCESSED_FILE, pickle_file = SCALER_MORDRED_FILE):
    
    from sklearn.preprocessing import StandardScaler
    import pickle
    train_drugs = list(pd.read_csv(train_drugs_file, sep = CSV_SEP, header = None).iloc[:,0])
    test_drugs = list(pd.read_csv(test_drugs_file, sep = CSV_SEP, header = None).iloc[:,0])
    opened_table = pd.read_csv(input_table_file, sep = CSV_SEP, header = 0)
    train_data = opened_table.loc[opened_table[mordred_drug_ID_col].isin(train_drugs)]
    scaler = StandardScaler()
    scaler.fit(train_data.drop([mordred_drug_ID_col], axis = 1))
    with open(pickle_file,'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    ids = list(opened_table[mordred_drug_ID_col])
    data = opened_table.drop([mordred_drug_ID_col], axis = 1)
    feature_columns = list(data)
    scaled_data = scaler.transform(data)
    output_table = pd.DataFrame(scaled_data, columns = feature_columns)
    output_table[mordred_drug_ID_col] = ids
    output_table.to_csv(output_file, sep = CSV_SEP, index = False)

class merge_dataset:

    def __init__(self, verbose = True, drug_1_col = "Drug1", \
                    drug_2_col = "Drug2", cell_col = "cell", \
                    mordred_file = MORDRED_PROCESSED_FILE, mordred_drug_ID_col = "NCI", \
                    logging_file = CCLE_DATASET_LOG_FILE):
        self.verbose = verbose
        self.drug1 = drug_1_col
        self.drug2 = drug_2_col
        self.cell = cell_col
        self.CCLE_subsets = ["5","6","7","12","14","17","22"]
        self.mordred_ID = mordred_drug_ID_col
        self.opened_drugs_features = pd.read_csv(mordred_file, sep = CSV_SEP, header = 0)

    def open_datasets(self, data_dictionary, id_class_columns):

        self.non_processable_columns = id_class_columns
        self.dataframes_dictionary = {}
        for current_key in list(data_dictionary.keys()):
            self.dataframes_dictionary[current_key] = pd.read_csv(data_dictionary[current_key], sep = CSV_SEP, header = 0)

    def merge_drugs(self):

        self.dataframes_drugs_dictionary = {} 
        for current_key in list(self.dataframes_dictionary.keys()):
            drug_1_dataframe = self.dataframes_dictionary[current_key].merge(self.opened_drugs_features, \
                        left_on = self.drug1, right_on = self.mordred_ID).drop([self.mordred_ID], axis = 1)
            self.dataframes_drugs_dictionary[current_key] = drug_1_dataframe.merge(self.opened_drugs_features, \
                        left_on = self.drug2, right_on = self.mordred_ID).drop([self.mordred_ID], axis = 1)
        del self.dataframes_dictionary

    def merge_CCLE(self, NA_handling = "drop", reduce_mode = "PCA", output_names_dictionary = {}):

        self.dataframes_CCLE_dictionary = {}
        for current_key in list(self.dataframes_drugs_dictionary.keys()):
            self.dataframes_CCLE_dictionary[current_key] = self.dataframes_drugs_dictionary[current_key]
            for current_subset in self.CCLE_subsets:
                subset_table = pd.read_csv(CCLE_FOLDER_PROCESSED + SYSTEM_SEP + reduce_mode + INTERMEDIATE_SEP + current_subset + ".csv")
                if NA_handling == "fill":
                    self.dataframes_CCLE_dictionary[current_key] = self.dataframes_CCLE_dictionary[current_key].merge(subset_table, on = self.cell, how = "left").fillna(0) 
                if NA_handling == "drop":
                    self.dataframes_CCLE_dictionary[current_key] = self.dataframes_CCLE_dictionary[current_key].merge(subset_table, on = self.cell, how = "left").dropna() 
                if self.verbose == True:
                    print("Merging CCLE, final shape:",self.dataframes_CCLE_dictionary[current_key].shape, "\n --For dataset:", current_key)
                self.dataframes_CCLE_dictionary[current_key].to_csv(output_names_dictionary[current_key], sep = CSV_SEP, index = False)
        del self.dataframes_drugs_dictionary

CCLE_processed = process_CCLE()
CCLE_processed.normalize_features()
CCLE_processed.reduce_CCLE(reduce_mode = "PCA", number_of_dimensions = 25)
CCLE_processed.reduce_CCLE(reduce_mode = "autoencoder")

normalize_drugs()

merge_object_1 = merge_dataset()
merge_object_1.open_datasets({"train": DATASETS_DICTIONARY["train_dataset"], \
                                    "test": DATASETS_DICTIONARY["test_dataset"], \
                                    "cell": DATASETS_DICTIONARY["independent_cell"], \
                                    "drugs": DATASETS_DICTIONARY["independent_drugs"], \
                                    "drug_combinations": DATASETS_DICTIONARY["independent_drug_combinations"]}, \
                                    id_class_columns = ["ZIP", "Loewe", "HSA", "Bliss", \
                                        "full_agreement", "block_id", "Drug1", "Drug2", "cell","full_agreement_val"])
merge_object_1.merge_drugs()
merge_object_1.merge_CCLE(reduce_mode = "PCA", NA_handling = "fill", \
                            output_names_dictionary = {"train": DATASETS_DICTIONARY["PCA_train_dataset_fillna"], \
                                                        "test": DATASETS_DICTIONARY["PCA_test_dataset_fillna"], \
                                                        "cell": DATASETS_DICTIONARY["PCA_independent_cell_fillna"], \
                                                        "drugs": DATASETS_DICTIONARY["PCA_independent_drugs_fillna"], \
                                                        "drug_combinations": DATASETS_DICTIONARY["PCA_independent_drug_combinations_fillna"]})

del merge_object_1
merge_object_2 = merge_dataset()
merge_object_2.open_datasets({"train": DATASETS_DICTIONARY["train_dataset"], \
                                    "test": DATASETS_DICTIONARY["test_dataset"], \
                                    "cell": DATASETS_DICTIONARY["independent_cell"], \
                                    "drugs": DATASETS_DICTIONARY["independent_drugs"], \
                                    "drug_combinations": DATASETS_DICTIONARY["independent_drug_combinations"]}, \
                                    id_class_columns = ["ZIP", "Loewe", "HSA", "Bliss", \
                                        "full_agreement", "block_id", "Drug1", "Drug2", "cell","full_agreement_val"])
merge_object_2.merge_drugs()
merge_object_2.merge_CCLE(reduce_mode = "autoencoder", NA_handling = "fill", \
                            output_names_dictionary = {"train": DATASETS_DICTIONARY["autoencoder_train_dataset_fillna"], \
                                                        "test": DATASETS_DICTIONARY["autoencoder_test_dataset_fillna"], \
                                                        "cell": DATASETS_DICTIONARY["autoencoder_independent_cell_fillna"], \
                                                        "drugs": DATASETS_DICTIONARY["autoencoder_independent_drugs_fillna"], \
                                                        "drug_combinations": DATASETS_DICTIONARY["autoencoder_independent_drug_combinations_fillna"]})
del merge_object_2
merge_object_3 = merge_dataset()
merge_object_3.open_datasets({"train": DATASETS_DICTIONARY["train_dataset"], \
                                    "test": DATASETS_DICTIONARY["test_dataset"], \
                                    "cell": DATASETS_DICTIONARY["independent_cell"], \
                                    "drugs": DATASETS_DICTIONARY["independent_drugs"], \
                                    "drug_combinations": DATASETS_DICTIONARY["independent_drug_combinations"]}, \
                                    id_class_columns = ["ZIP", "Loewe", "HSA", "Bliss", \
                                        "full_agreement", "block_id", "Drug1", "Drug2", "cell","full_agreement_val"])
merge_object_3.merge_drugs()
merge_object_3.merge_CCLE(reduce_mode = "PCA", NA_handling = "drop", \
                            output_names_dictionary = {"train": DATASETS_DICTIONARY["PCA_train_dataset_dropna"], \
                                                        "test": DATASETS_DICTIONARY["PCA_test_dataset_dropna"], \
                                                        "cell": DATASETS_DICTIONARY["PCA_independent_cell_dropna"], \
                                                        "drugs": DATASETS_DICTIONARY["PCA_independent_drugs_dropna"], \
                                                        "drug_combinations": DATASETS_DICTIONARY["PCA_independent_drug_combinations_dropna"]})

del merge_object_3
merge_object_4 = merge_dataset()
merge_object_4.open_datasets({"train": DATASETS_DICTIONARY["train_dataset"], \
                                    "test": DATASETS_DICTIONARY["test_dataset"], \
                                    "cell": DATASETS_DICTIONARY["independent_cell"], \
                                    "drugs": DATASETS_DICTIONARY["independent_drugs"], \
                                    "drug_combinations": DATASETS_DICTIONARY["independent_drug_combinations"]}, \
                                    id_class_columns = ["ZIP", "Loewe", "HSA", "Bliss", \
                                        "full_agreement", "block_id", "Drug1", "Drug2", "cell","full_agreement_val"])
merge_object_4.merge_drugs()
merge_object_4.merge_CCLE(reduce_mode = "autoencoder", NA_handling = "drop", \
                            output_names_dictionary = {"train": DATASETS_DICTIONARY["autoencoder_train_dataset_dropna"], \
                                                        "test": DATASETS_DICTIONARY["autoencoder_test_dataset_dropna"], \
                                                        "cell": DATASETS_DICTIONARY["autoencoder_independent_cell_dropna"], \
                                                        "drugs": DATASETS_DICTIONARY["autoencoder_independent_drugs_dropna"], \
                                                        "drug_combinations": DATASETS_DICTIONARY["autoencoder_independent_drug_combinations_dropna"]})
del merge_object_4