#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variables for the Synpred code
"""

__author__ = "A.J. Preto"
__email__ = "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "SynPred"

"""
Global variables
"""
CSV_SEP = ","
SECONDARY_CSV_SEP = ";"
PARAGRAPH_SEP = "\n"
SYSTEM_SEP = "/"
INTERMEDIATE_SEP = "_"
TAB_SEP = "\t"
CSV_TERMINATION = ".csv"
SDF_TERMINATION = ".sdf"
PKL_TERMINATION = ".pkl"
TXT_TERMINATION = ".txt"
SMILE_TERMINATION = ".smile"
DEFAULT_LOCATION = "/insert/your/path"
DATASET_FOLDER =  DEFAULT_LOCATION + SYSTEM_SEP + "datasets"
SPLIT_TABLES_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "split_tables"
CCLE_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "CCLE"
CCLE_FOLDER_PROCESSED = DEFAULT_LOCATION + SYSTEM_SEP + "CCLE_processed"
SUPPORT_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "support"
FEATURE_IMPORTANCE_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "feature_importance"
MOLECULES_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "molecules"
REDEPLOYMENT_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "redeployment_variables"
DL_SAVED_MODELS = DEFAULT_LOCATION + SYSTEM_SEP + "saved_model"
EVALUATION_NON_DL_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "evaluation_non_DL"
EVALUATION_DL_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "evaluation_summary"
EVALUATION_DL_INDEX = "gridsearch_index" + CSV_TERMINATION
EVALUATION_DL_CUSTOM_INDEX = "gridsearch_index_custom" + CSV_TERMINATION
RESOURCES_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "resources"
ML_GRIDSEARCH_RESULTS = SUPPORT_FOLDER + SYSTEM_SEP + "ML_gridsearch.csv"
ML_GRIDSEARCH_PARAMETERS = SUPPORT_FOLDER + SYSTEM_SEP + "ML_gridsearch_parameters.csv"
DL_GRIDSEARCH_RESULTS = SUPPORT_FOLDER + SYSTEM_SEP + "DL_gridsearch_index.csv"
DL_GRIDSEARCH_PARAMETERS = SUPPORT_FOLDER + SYSTEM_SEP + "DL_gridsearch_parameters.csv"
DL_ENSEMBLE_PARAMETERS = SUPPORT_FOLDER + SYSTEM_SEP + "DL_ensemble_parameters.csv"
CONC_BEST_PARAMETERS_INDEX_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "gridsearch_index_after_grid_conc.csv"
BEST_PARAMETERS_KERAS_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "gridsearch_index" + CSV_TERMINATION
BEST_PARAMETERS_INDEX_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "gridsearch_index_after_grid" + CSV_TERMINATION
CONCENTRATION_SCALING_FILE = SUPPORT_FOLDER + SYSTEM_SEP  + "concentration_scaling_train" + CSV_TERMINATION

"""
NCI-ALMANAC variables
"""

#NCI_ALMANAC_ALL_CLASSES = DATASET_FOLDER + SYSTEM_SEP + "NCI_ALMANAC_drug_combinations" + CSV_TERMINATION
NCI_ALMANAC_CELL_COLUMN = "CELLNAME"
CLASS_TRAIN_DATASET = DATASET_FOLDER + SYSTEM_SEP + "train_classes" + CSV_TERMINATION
CLASS_TEST_DATASET = DATASET_FOLDER + SYSTEM_SEP + "test_classes" + CSV_TERMINATION 

DATASETS_DICTIONARY = {"NCI_ALMANAC": DATASET_FOLDER + SYSTEM_SEP + "NCI_ALMANAC" + CSV_TERMINATION, \
			"NCI_ALMANAC_synfinder": DATASET_FOLDER + SYSTEM_SEP + "calculated_class_NCI_ALMANAC" + CSV_TERMINATION, \
			"NCI_ALMANAC_classes": DATASET_FOLDER + SYSTEM_SEP + "NCI_ALMANAC_no_concentrations" + CSV_TERMINATION, \
			"NCI_ALMANAC_comboscore" : DATASET_FOLDER + SYSTEM_SEP + "comboscore_processed_NCI_ALMANAC" + CSV_TERMINATION, \
			"NCI_ALMANAC_with_combinations": DATASET_FOLDER + SYSTEM_SEP + "NCI_ALMANAC_drug_combinations" + CSV_TERMINATION, \

			"train_dataset": DATASET_FOLDER + SYSTEM_SEP + "train" + CSV_TERMINATION, \
			"test_dataset": DATASET_FOLDER + SYSTEM_SEP + "test" + CSV_TERMINATION, \
			"independent_cell": DATASET_FOLDER + SYSTEM_SEP + "independent_cells" + CSV_TERMINATION, \
			"independent_drugs": DATASET_FOLDER + SYSTEM_SEP + "independent_drugs" + CSV_TERMINATION, \
			"independent_drug_combinations": DATASET_FOLDER + SYSTEM_SEP + "independent_drug_combinations" + CSV_TERMINATION, \

			"train_concentration_dataset": DATASET_FOLDER + SYSTEM_SEP + "train_concentration" + CSV_TERMINATION, \
			"test_concentration_dataset": DATASET_FOLDER + SYSTEM_SEP + "test_concentration" + CSV_TERMINATION, \
			"independent_cell_concentration": DATASET_FOLDER + SYSTEM_SEP + "independent_cells_concentration" + CSV_TERMINATION, \
			"independent_drugs_concentration": DATASET_FOLDER + SYSTEM_SEP + "independent_drugs_concentration" + CSV_TERMINATION, \
			"independent_drug_combinations_concentration": DATASET_FOLDER + SYSTEM_SEP + "independent_drug_combinations_concentration" + CSV_TERMINATION, \
			
			"PCA_train_dataset_dropna": DATASET_FOLDER + SYSTEM_SEP + "PCA_train_dropna" + CSV_TERMINATION, \
			"PCA_test_dataset_dropna": DATASET_FOLDER + SYSTEM_SEP + "PCA_test_dropna" + CSV_TERMINATION, \
			"PCA_independent_cell_dropna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_cell_dropna" + CSV_TERMINATION, \
			"PCA_independent_drugs_dropna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_drugs_dropna" + CSV_TERMINATION, \
			"PCA_independent_drug_combinations_dropna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_drug_combinations_dropna" + CSV_TERMINATION, \

			"autoencoder_train_dataset_dropna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_train_dropna" + CSV_TERMINATION, \
			"autoencoder_test_dataset_dropna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_test_dropna" + CSV_TERMINATION, \
			"autoencoder_independent_cell_dropna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_cell_dropna" + CSV_TERMINATION, \
			"autoencoder_independent_drugs_dropna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_drugs_dropna" + CSV_TERMINATION, \
			"autoencoder_independent_drug_combinations_dropna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_drug_combinations_dropna" + CSV_TERMINATION, \
			
			"PCA_train_dataset_fillna": DATASET_FOLDER + SYSTEM_SEP + "PCA_train_fillna" + CSV_TERMINATION, \
			"PCA_test_dataset_fillna": DATASET_FOLDER + SYSTEM_SEP + "PCA_test_fillna" + CSV_TERMINATION, \
			"PCA_independent_cell_fillna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_cell_fillna" + CSV_TERMINATION, \
			"PCA_independent_drugs_fillna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_drugs_fillna" + CSV_TERMINATION, \
			"PCA_independent_drug_combinations_fillna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_drug_combinations_fillna" + CSV_TERMINATION, \

			"autoencoder_train_dataset_fillna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_train_fillna" + CSV_TERMINATION, \
			"autoencoder_test_dataset_fillna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_test_fillna" + CSV_TERMINATION, \
			"autoencoder_independent_cell_fillna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_cell_fillna" + CSV_TERMINATION, \
			"autoencoder_independent_drugs_fillna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_drugs_fillna" + CSV_TERMINATION, \
			"autoencoder_independent_drug_combinations_fillna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_drug_combinations_fillna" + CSV_TERMINATION, \

			"PCA_train_concentration_dataset_dropna": DATASET_FOLDER + SYSTEM_SEP + "PCA_train_concentration_dropna" + CSV_TERMINATION, \
			"PCA_test_concentration_dataset_dropna": DATASET_FOLDER + SYSTEM_SEP + "PCA_test_concentration_dropna" + CSV_TERMINATION, \
			"PCA_independent_cell_concentration_dropna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_cell_concentration_dropna" + CSV_TERMINATION, \
			"PCA_independent_drugs_concentration_dropna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_drugs_concentration_dropna" + CSV_TERMINATION, \
			"PCA_independent_drug_combinations_concentration_dropna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_drug_combinations_concentration_dropna" + CSV_TERMINATION, \

			"autoencoder_train_concentration_dataset_dropna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_train_concentration_dropna" + CSV_TERMINATION, \
			"autoencoder_test_concentration_dataset_dropna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_test_concentration_dropna" + CSV_TERMINATION, \
			"autoencoder_independent_cell_concentration_dropna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_cell_concentration_dropna" + CSV_TERMINATION, \
			"autoencoder_independent_drugs_concentration_dropna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_drugs_concentration_dropna" + CSV_TERMINATION, \
			"autoencoder_independent_drug_combinations_concentration_dropna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_drug_combinations_concentration_dropna" + CSV_TERMINATION, \
			
			"PCA_train_concentration_dataset_fillna": DATASET_FOLDER + SYSTEM_SEP + "PCA_train_concentration_fillna" + CSV_TERMINATION, \
			"PCA_test_concentration_dataset_fillna": DATASET_FOLDER + SYSTEM_SEP + "PCA_test_concentration_fillna" + CSV_TERMINATION, \
			"PCA_independent_cell_concentration_fillna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_cell_concentration_fillna" + CSV_TERMINATION, \
			"PCA_independent_drugs_concentration_fillna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_drugs_concentration_fillna" + CSV_TERMINATION, \
			"PCA_independent_drug_combinations_concentration_fillna": DATASET_FOLDER + SYSTEM_SEP + "PCA_independent_drug_combinations_concentration_fillna" + CSV_TERMINATION, \

			"autoencoder_train_concentration_dataset_fillna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_train_concentration_fillna" + CSV_TERMINATION, \
			"autoencoder_test_concentration_dataset_fillna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_test_concentration_fillna" + CSV_TERMINATION, \
			"autoencoder_independent_cell_concentration_fillna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_cell_concentration_fillna" + CSV_TERMINATION, \
			"autoencoder_independent_drugs_concentration_fillna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_drugs_concentration_fillna" + CSV_TERMINATION, \
			"autoencoder_independent_drug_combinations_concentration_fillna": DATASET_FOLDER + SYSTEM_SEP + "autoencoder_independent_drug_combinations_concentration_fillna" + CSV_TERMINATION
			
			}
DATASETS_DICTIONARY_NO_CONCENTRATION = {"PCA_fillna": [DATASETS_DICTIONARY["PCA_train_dataset_fillna"], DATASETS_DICTIONARY["PCA_test_dataset_fillna"]], \
                        "PCA_dropna": [DATASETS_DICTIONARY["PCA_train_dataset_dropna"], DATASETS_DICTIONARY["PCA_test_dataset_dropna"]], \
                        "autoencoder_fillna": [DATASETS_DICTIONARY["autoencoder_train_dataset_fillna"], DATASETS_DICTIONARY["autoencoder_test_dataset_fillna"]], \
                        "autoencoder_dropna": [DATASETS_DICTIONARY["autoencoder_train_dataset_dropna"], DATASETS_DICTIONARY["autoencoder_test_dataset_dropna"]]}
     
DATASETS_DICTIONARY_CONCENTRATION = {"PCA_fillna": [DATASETS_DICTIONARY["PCA_train_concentration_dataset_fillna"], DATASETS_DICTIONARY["PCA_test_concentration_dataset_fillna"]], \
                        "PCA_dropna": [DATASETS_DICTIONARY["PCA_train_concentration_dataset_dropna"], DATASETS_DICTIONARY["PCA_test_concentration_dataset_dropna"]], \
                        "autoencoder_fillna": [DATASETS_DICTIONARY["autoencoder_train_concentration_dataset_fillna"], DATASETS_DICTIONARY["autoencoder_test_concentration_dataset_fillna"]], \
                        "autoencoder_dropna": [DATASETS_DICTIONARY["autoencoder_train_concentration_dataset_dropna"], DATASETS_DICTIONARY["autoencoder_test_concentration_dataset_dropna"]]}

DATASET_H5_TRAIN_PCA_FILLNA = DATASET_FOLDER + SYSTEM_SEP + "PCA_train_concentration_fillna.h5"
H5_HEADER_PCA_FILLNA = SUPPORT_FOLDER + SYSTEM_SEP + "concentration_header.pkl"

COLUMN_CLASSES = ["ZIP","Bliss","HSA", "Loewe"]
FULL_AGREEMENT_COLUMN_NAME = "full_agreement"

SCALER_MORDRED_FILE = "scaler_mordred.pkl"
SCALER_MORDRED_FILE_ONE_HOT = "scaler_mordred_one_hot.pkl"
SCALER_CUSTOM_FILE = "scaler_custom.pkl"

"""
h5 datasets
"""
H5_DATASET_RAW = DATASET_FOLDER + SYSTEM_SEP + "dataset_raw.h5"

"""
CCLE variables
"""
SCALED_CCLE_START = "scaler_CCLE_subset"
CCLE_ID_COLUMN_NAME = "ID"
CCLE_ID_COLUMN_SEP = "&"
CCLE_ANNOTATION_FILE = CCLE_FOLDER + SYSTEM_SEP + "Cell_line_annotations.txt"
SUBSET_CCLE = "CCLE_subset" + CSV_TERMINATION
FILTERED_CCLE = "CCLE_filtered" + CSV_TERMINATION
PCA_CCLE = "CCLE_PCA" + CSV_TERMINATION
TSNE_CCLE = "CCLE_tsne" + CSV_TERMINATION
AUTOENCODER_CCLE = "CCLE_autoencoder" + CSV_TERMINATION
CCLE_COLUMN_TAG = "CCLE"

"""
Support variables
"""
CCLE_DATASET_LOG_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "CCLE_log_file" + CSV_TERMINATION
CCLE_COLUMNS_KEPT_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "columns_kept_names" + CSV_TERMINATION
NCI_DRUG_CONVERTER_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "NCIOPENB_SMI" + TXT_TERMINATION
FEATURES_IMPORTANCE_OUTPUT_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "feature_importance.csv"
TSNE_PLOT_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "tsne_plot.png"
TSNE_TABLE_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "tsne_table.csv"
PCA_PLOT_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "PCA_plot.png"
PCA_TABLE_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "PCA_table.csv"
AUTOENCODER_LOG_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "autoencoder_scores.csv"
MORDRED_RAW_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "mordred_features_clean.csv"
MORDRED_PROCESSED_FILE = SUPPORT_FOLDER + SYSTEM_SEP + "mordred_features_normalized.csv"
CORRECTION_DICTIONARY = {"CAKI-1":"Caki-1","NCI-H322M":"NCI-H322",
		"MDA-MB-231/ATCC":"MDA-MB-231","A498":"A-498","OVCAR-3":"NIH:OVCAR-3",
		"HS 578T":"Hs 578T","MDA-MB-435":"MDA-MB-435S","786-0":"786-O",
		"MALME-3M":"Malme-3M","HL-60(TB)":"HL-60","SW-620":"SW620",
		"HCT-116":"HCT 116","DU-145":"DU 145","A549/ATCC":"A549",
		"U251":"U-251 MG","RPMI-8226":"RPMI 8226","HT29":"HT-29",
		"SR":"SR-786","TK-10":"TK10","SF-268":"SF268",
		"SF-539":"SF539","SNB-75":"SNB75","HCC-2998":"HCC2998",
		"NCI/ADR-RES":"ADRRES","RXF 393":"RXF393","SN12C":"SN-12C",
		"CCRF-CEM":"CCRFCEM"}
DROPPABLE_COLUMNS = ['cell', 'Drug1', 'Drug2', 'ZIP', 'Bliss', 'HSA', 'Loewe','full_agreement',"combo_score","conc1","conc2","block_id","full_agreement_val"]
BLOCK_ID_COL = "block_id"
TARGET_CLASS_COLUMN = ["full_agreement"]
RANDOM_STATE = 42
METRICS_NAME_LIST = ["Accuracy","Precision","Recall","AUC","F1-score"]
CLASSES_LIST = ["ZIP","Bliss","HSA","Loewe"]
CELL_TYPES = {
        'Brain': ['U251', 'SF-268', 'SF-295', 'SNB-75', 'SF-539'],
        'Breast': ['MDA-MB-231/ATCC', 'MDA-MB-468', 'T-47D', 'MCF7', 'BT-549', 'HS 578T'],
        'Colon': ['HCT-15', 'SW-620', 'HCT-116', 'KM12'],
        'Haematological': ['SR', 'K-562', 'RPMI-8226,'],
        'Kidney': ['CAKI-1', 'ACHN', 'UO-31', 'A498', '786-0'],
        'Lung': ['NCI-H460', 'EKVX', 'HOP-62', 'A549/ATCC', 'NCI-H23', 'NCI-H522', 'NCI-H322M', 'HOP-92'],
        'Ovary': ['OVCAR-8', 'OVCAR-3', 'OVCAR-4', 'IGROV1', 'SK-OV-3'],
        'Prostate': ['DU-145', 'PC-3'],
        'Skin': ['LOX IMVI', 'SK-MEL-28', 'SK-MEL-5', 'MALME-3M', 'UACC-62', 'UACC-257']
    }
METRICS_CLASSIFICATION = ["Accuracy","Precision","Recall","AUC","F1-value"]
METRICS_REGRESSION = ["RMSE","MSE","Pearson","r^2","MAE","Spearman"]
POSSIBLE_TARGETS = ["Loewe","Bliss","ZIP","HSA","full_agreement"]
CONC_CLASSES_LIST = ["ZIP","Bliss","HSA","Loewe","combo_score"]
CONC_POSSIBLE_TARGETS = ["Loewe","Bliss","ZIP","HSA","combo_score","full_agreement"]
CONC_DROPPABLE_COLUMNS = ['cell', 'Drug1', 'Drug2', 'ZIP', 'Bliss', 'HSA', 'Loewe','full_agreement',"combo_score","block_id","full_agreement_val"]