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
                            SCALED_CCLE_START, H5_DATASET_RAW

from synpred_variables import DEFAULT_LOCATION, CCLE_FOLDER, \
                            SUPPORT_FOLDER, CCLE_DATASET_LOG_FILE, \
                            REDEPLOYMENT_FOLDER, CCLE_ID_COLUMN_NAME, \
                            TRAIN_DATASET, TEST_DATASET, CCLE_ID_COLUMN_SEP, \
                            SUBSET_CCLE, FILTERED_CCLE, CCLE_ANNOTATION_FILE, \
                            NCI_ALMANAC_DATASET, NCI_ALMANAC_CELL_COLUMN, \
                            TRAIN_DATASET_RAW, TEST_DATASET_RAW, PCA_CCLE, \
                            CCLE_COLUMN_TAG, TRAIN_DATASET_PCA, \
                            TEST_DATASET_PCA, TRAIN_DATASET_PROCESSED, \
                            TEST_DATASET_PROCESSED, AUTOENCODER_CCLE, \
                            AUTOENCODER_LOG_FILE, TRAIN_DATASET_AUTOENCODER, \
                            TEST_DATASET_AUTOENCODER, TRAIN_DATASET_AUTOENCODER_DROP, \
                            TEST_DATASET_AUTOENCODER_DROP, TRAIN_DATASET_PCA_DROP, \
                            TEST_DATASET_PCA_DROP, RANDOM_STATE

from synpred_support_functions import open_log_file, alternative_ID_file
import os
import pandas as pd
import tensorflow as tf
import sys
import numpy as np
import pickle
import h5py as h5
import random

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def CCLE_subset_id_expand(input_table_row):

    """
    Alternative IDs for each CCLE table row
    """
    current_id = input_table_row[0]
    current_id_split = current_id.split(CCLE_ID_COLUMN_SEP)
    if len(current_id_split) > 2:
        CCLE_subset_id_list = current_id_split[1:]
        CCLE_subset_id_list = CCLE_subset_id_list + [CCLE_subset_id_list[0].split(INTERMEDIATE_SEP)[0], CCLE_subset_id_list[0].split(INTERMEDIATE_SEP)[1]]
    else:
        CCLE_subset_id_list = [current_id] + [current_id.split(INTERMEDIATE_SEP)[0]] + [current_id.split(INTERMEDIATE_SEP)[1]]
    return CCLE_subset_id_list

def compare_id_lists(input_dataset_list, input_CCLE_list):

    """
    Return True or False depending on whether or not there is a match for the lists
    """
    for x in input_dataset_list:
        for y in input_CCLE_list:
            if x == y: return True
    return False

def get_template_tables(logger_file, alternative_ids_dictionary, get_mode = "PCA"):

    """
    Get the template tables with a new column adding the matching ID
    """
    output_dictionary = {}
    for current_subset in logger_file.keys():
        if get_mode == "PCA":
            subset_file_name = CCLE_FOLDER + SYSTEM_SEP + current_subset + INTERMEDIATE_SEP + PCA_CCLE
        elif get_mode == "autoencoder":
            subset_file_name = CCLE_FOLDER + SYSTEM_SEP + current_subset + INTERMEDIATE_SEP + AUTOENCODER_CCLE
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

def generate_raw_dataset(input_dataset, output_name, target_folder, \
                            logging_file = CCLE_DATASET_LOG_FILE, mode = "PCA", \
                            NA_handling = "drop"):

    """
    Gather the subsets for raw CCLE files and join with the train and test datasets
    
    """
    logger = open_log_file(logging_file)
    major_file = pd.read_csv(input_dataset, header = 0)
    alternative_dictionary = alternative_ID_file()
    tables_dictionary = get_template_tables(logger, alternative_dictionary, get_mode = mode)
    for current_subset in logger.keys():
        print("Currently evaluating subset", current_subset)
        subset_table = tables_dictionary[current_subset].iloc[:,1:].add_prefix(CCLE_COLUMN_TAG + INTERMEDIATE_SEP + current_subset + INTERMEDIATE_SEP)
        subset_table = reconsider_id_column(subset_table.rename(columns = {CCLE_COLUMN_TAG + INTERMEDIATE_SEP + current_subset + INTERMEDIATE_SEP + "cell": "cell"}))
        try:
            subset_table = subset_table.drop(["Unnamed: 0.1"], axis = 1)
        except:
            pass
        columns_list = list(major_file) + list(subset_table)[:-1]
        if NA_handling == "fill":
            major_file = major_file.merge(subset_table, on = "cell", how = "left").fillna(0)
        if NA_handling == "drop":
            major_file = major_file.merge(subset_table, on = "cell", how = "left").dropna()
        major_file.columns = columns_list
        print("Final shape:",major_file.shape)
    major_file.to_csv(output_name, index = False)

def autoencoder(input_table, input_size, input_architecture, subset_name, \
                write_mode = True, number_of_epochs = 1000):

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LeakyReLU, \
                                    BatchNormalization
    import tensorflow as tf
    """
    Perform dimensionality reduction with autoencoder
    """
    input_layer = Input(shape=(input_size,))
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
    output = Dense(input_size, activation='linear')(decoder)
    # define autoencoder model
    model = Model(inputs=input_layer, outputs=output)
    # compile autoencoder model
    model.compile(optimizer="adam", loss='mse')
    model.fit(input_table, input_table, epochs = number_of_epochs, verbose = True)

    dimensionality_model = Model(inputs=input_layer, outputs=bottleneck)
    intermediate_table = model.predict(input_table)
    output_table = dimensionality_model.predict(input_table)

    print("RMSE:",np.mean(np.sqrt(np.mean((intermediate_table-input_table)**2))))
    if write_mode == True:
        with open(AUTOENCODER_LOG_FILE, "a") as log_file:
            to_write_row = subset_name + CSV_SEP + \
                str(np.mean(np.sqrt(np.mean((intermediate_table-input_table)**2)))) + PARAGRAPH_SEP
            log_file.write(to_write_row)
    return output_table

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

def generate_reduced_subsets(logging_file = CCLE_DATASET_LOG_FILE, \
                            number_of_dimensions = 50, \
                            verbose = True, mode = "PCA", \
                            write_feature_importances = False):

    """
    Generate PCA reduced CCLE subsets
    """
    from sklearn.decomposition import PCA
    import pickle
    logger = open_log_file(logging_file)
    for current_subset in logger.keys():
        subset_file_name = CCLE_FOLDER + SYSTEM_SEP + current_subset + INTERMEDIATE_SEP + FILTERED_CCLE
        opened_file = pd.read_csv(subset_file_name)
        try:
            ID_column = opened_file["Unnamed: 0"]
            features_columns = opened_file.drop(["Unnamed: 0"], axis = 1)
        except:
            ID_column = opened_file[CCLE_ID_COLUMN_NAME]
            features_columns = opened_file.drop([CCLE_ID_COLUMN_NAME], axis = 1)
        if mode == "PCA":
            output_name = CCLE_FOLDER + SYSTEM_SEP + current_subset + INTERMEDIATE_SEP + PCA_CCLE
            pca = PCA(n_components = number_of_dimensions)
            pca.fit(features_columns)
            if write_feature_importances == True:
                output_importance_name = SUPPORT_FOLDER + SYSTEM_SEP + current_subset + \
                            INTERMEDIATE_SEP + "importance" + INTERMEDIATE_SEP + PCA_CCLE
                output_variance_name = SUPPORT_FOLDER + SYSTEM_SEP + current_subset + \
                            INTERMEDIATE_SEP + "variance" + INTERMEDIATE_SEP + PCA_CCLE
                output_PCA_name = SUPPORT_FOLDER + SYSTEM_SEP + current_subset + \
                            INTERMEDIATE_SEP + "PCA_transform.pkl"
                with open(output_PCA_name,'wb') as f:
                    pickle.dump(pca,f)
                if verbose == True:
                    print("Currently writing:", output_importance_name)
                explain_PCA_variance(pca, output_importance_name, features_columns, number_of_dimensions, output_variance_name)

            output_features_table = pca.transform(features_columns)
        if mode == "autoencoder":
            output_name = CCLE_FOLDER + SYSTEM_SEP + current_subset + INTERMEDIATE_SEP + AUTOENCODER_CCLE
            features_size = features_columns.shape[1]
            sample_size = features_columns.shape[0]
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
            output_features_table = autoencoder(features_columns, features_size, architecture, current_subset, number_of_epochs = epochs)
        output_table = pd.concat([ID_column, pd.DataFrame(output_features_table)], axis = 1)
        output_table.to_csv(output_name, index = False)
        if verbose == True:
            print("Successfully reduced", subset_file_name.split(SYSTEM_SEP)[-1], "with a", mode ," from", \
                opened_file.shape, "to a shape of:",output_table.shape)

"""
First generate the reduced subsets
- activate write_feature_importances to write feature contribution to each component
"""
#generate_reduced_subsets(number_of_dimensions = 25, mode = "autoencoder")
generate_reduced_subsets(number_of_dimensions = 25, mode = "PCA", write_feature_importances = True)
"""
Then aggregate them with full dataset
"""

#generate_reduced_subsets(number_of_dimensions = 25, mode = "PCA", write_feature_importances = True)
#sys.exit()

"""
generate_raw_dataset(TRAIN_DATASET_PROCESSED, TRAIN_DATASET_AUTOENCODER_DROP, \
                        CCLE_FOLDER, mode = "autoencoder", NA_handling = "drop")
generate_raw_dataset(TEST_DATASET_PROCESSED, TEST_DATASET_AUTOENCODER_DROP, \
                        CCLE_FOLDER, mode = "autoencoder", NA_handling = "drop")

generate_raw_dataset(TRAIN_DATASET_PROCESSED, TRAIN_DATASET_AUTOENCODER, \
                        CCLE_FOLDER, mode = "autoencoder", NA_handling = "fill")
generate_raw_dataset(TEST_DATASET_PROCESSED, TEST_DATASET_AUTOENCODER, \
                        CCLE_FOLDER, mode = "autoencoder", NA_handling = "fill")
"""
generate_raw_dataset(TRAIN_DATASET_PROCESSED, TRAIN_DATASET_PCA_DROP, \
                        CCLE_FOLDER, mode = "PCA", NA_handling = "drop")
generate_raw_dataset(TEST_DATASET_PROCESSED, TEST_DATASET_PCA_DROP, \
                        CCLE_FOLDER, mode = "PCA", NA_handling = "drop")

generate_raw_dataset(TRAIN_DATASET_PROCESSED, TRAIN_DATASET_PCA, \
                        CCLE_FOLDER, mode = "PCA", NA_handling = "fill")
generate_raw_dataset(TEST_DATASET_PROCESSED, TEST_DATASET_PCA, \
                        CCLE_FOLDER, mode = "PCA", NA_handling = "fill")

