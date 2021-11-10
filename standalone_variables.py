"""
List all usable variables
"""

def process_cells_file(input_file, mode = "dropdown", \
                        target_column = "", input_cell_type = ""):

    import pandas as pd
    opened_file = pd.read_csv(input_file, sep = SEP, header = 0)
    if mode == "dropdown":
        return list(opened_file[target_column].unique())
    if mode == "features_extraction":
        subset_table = opened_file.loc[opened_file[target_column] == input_cell_type][CELL_LINES_ID]
        if subset_table.empty == True:
            subset_table = opened_file.loc[opened_file["GDSC"] == input_cell_type][CELL_LINES_ID]
        #print(subset_table, input_cell_type)
        #print(opened_file)
        return list(subset_table)

"""
Folder paths
"""
HOME = "/insert/your/path"
SYSTEM_SEP = "/"
TEMPLATES = HOME + SYSTEM_SEP + "templates"
UPLOAD_FOLDER = HOME + SYSTEM_SEP + "uploads"
ONLY_UPLOAD_FOLDER = "uploads"
SUPPORT_FOLDER = "support"
RESOURCES_FOLDER = "resources"
DATA_FOLDER = "data"
GRAPH_STATIC_FOLDER = HOME + SYSTEM_SEP + "static/img/temp_graph"
ML_MODELS_FOLDER = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + "ML_models"
DL_MODELS_FOLDER = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + "DL_models"

REFERENCE_MODELS_LIST = ["Full-agreement", "Bliss", "ZIP", "HSA", "Loewe"]
MODELS_DICTIONARY = {
    "Full-agreement": [ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_ETC_Full-agreement.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_SVM_Full-agreement.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_SGD_Full-agreement.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_RF_Full-agreement.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_XGB_Full-agreement.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_KNN_Full-agreement.pkl", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "1_after_grid_save_full_agreement.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "4_after_grid_save_full_agreement.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "2_after_grid_save_full_agreement.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "3_after_grid_save_full_agreement.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "5_after_grid_save_full_agreement.h5"
        ], \
    "Bliss": [ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_RF_Bliss.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_SVM_Bliss.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_ETC_Bliss.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_XGB_Bliss.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_KNN_Bliss.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_SGD_Bliss.pkl", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "15_after_grid_save_Bliss.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "14_after_grid_save_Bliss.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "16_after_grid_save_Bliss.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "17_after_grid_save_Bliss.h5"
        ], \
    "Loewe": [ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_SVM_Loewe.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_RF_Loewe.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_XGB_Loewe.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_SGD_Loewe.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_KNN_Loewe.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_ETC_Loewe.pkl", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "8_after_grid_save_Loewe.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "10_after_grid_save_Loewe.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "6_after_grid_save_Loewe.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "12_after_grid_save_Loewe.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "11_after_grid_save_Loewe.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "9_after_grid_save_Loewe.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "7_after_grid_save_Loewe.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "13_after_grid_save_Loewe.h5"
        ], \
    "ZIP": [ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_SGD_ZIP.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_KNN_ZIP.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_SVM_ZIP.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_RF_ZIP.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_ETC_ZIP.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_XGB_ZIP.pkl", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "22_after_grid_save_ZIP.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "24_after_grid_save_ZIP.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "25_after_grid_save_ZIP.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "26_after_grid_save_ZIP.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "23_after_grid_save_ZIP.h5"
        ], \
    "HSA": [ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_ETC_HSA.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_RF_HSA.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_SVM_HSA.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_XGB_HSA.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_KNN_HSA.pkl", \
        ML_MODELS_FOLDER + SYSTEM_SEP + "PCA_fillna_SGD_HSA.pkl", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "19_after_grid_save_HSA.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "18_after_grid_save_HSA.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "20_after_grid_save_HSA.h5", \
        DL_MODELS_FOLDER + SYSTEM_SEP + "21_after_grid_save_HSA.h5"
        ]
}

ENSEMBLE_MODELS_DICTIONARY = {"Full-agreement": DL_MODELS_FOLDER + SYSTEM_SEP + "full_agreement_final_ensemble.h5", \
        "HSA": DL_MODELS_FOLDER + SYSTEM_SEP + "HSA_final_ensemble.h5", \
        "ZIP": DL_MODELS_FOLDER + SYSTEM_SEP + "ZIP_final_ensemble.h5", \
        "Loewe": DL_MODELS_FOLDER + SYSTEM_SEP + "Loewe_final_ensemble.h5", \
        "Bliss": DL_MODELS_FOLDER + SYSTEM_SEP + "Bliss_final_ensemble.h5"}
"""
HSA:
1-ETC
2-RF
3-SVM
4-XGB
5-KNN
6-SGD
7-DL_19
8-DL_18
9-DL_20
10-DL_21

ZIP:
1-SGD
2-KNN
3-SVM
4-RF
5-ETC
6-XGB
7-DL_22
8-DL_24
9-DL_25
10-DL_26
11-DL_23

Loewe:
1-SVM
2-RF
3-XGB
4-SGD
5-KNN
6-ETC
7-DL_8
8-DL_10
9-DL_6
10-DL_12
11-DL_11
12-DL_9
13-DL_7
14-DL_13

Bliss:
1-RF
2-SVM
3-ETC
4-XGB
5-KNN
6-SGD
7-DL_15
8-DL_14
9-DL_16
10-DL_17

Full_agreement:
1-ETC
2-SVM
3-SGD
4-RF
5-XGB
6-KNN
7-DL_1
8-DL_4
9-DL_2
10-DL_3
11-DL_5

"""
TENSORFLOW_CHECKPOINT = HOME + SYSTEM_SEP + SUPPORT_FOLDER
SEP = ","
INTERMEDIATE_SEP = "_"
TXT_TERMINATION = ".txt"
CSV_TERMINATION = ".csv"
PROCESSED_TERMINATION = INTERMEDIATE_SEP + "processed" + CSV_TERMINATION
FEATURES_TERMINATION = INTERMEDIATE_SEP + "features" + CSV_TERMINATION
PREDICTION_COL_NAME = "Full-agreement"

"""
Cell lines files
"""
CELL_LINES_FILE = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + DATA_FOLDER + SYSTEM_SEP + "almanac_cell_lines_library.csv"
CELL_LINES_ID = "NCI-60"
CELL_LINES_TYPE = "TISSUE"
CELL_LINES_COLUMN = "cells"
CELL_LINES_CNV = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + DATA_FOLDER + SYSTEM_SEP + "dimred_cnv_final.csv"
CELL_LINES_RNASEQ = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + DATA_FOLDER + SYSTEM_SEP + "dimred_rnaseq_final.csv"
CELL_LINES_MET = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + DATA_FOLDER + SYSTEM_SEP + "dimred_rrbs_final.csv"
CELL_LINES_5 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + DATA_FOLDER + SYSTEM_SEP + "PCA_5.csv"
CELL_LINES_6 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + DATA_FOLDER + SYSTEM_SEP + "PCA_6.csv"
CELL_LINES_7 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + DATA_FOLDER + SYSTEM_SEP + "PCA_7.csv"
CELL_LINES_12 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + DATA_FOLDER + SYSTEM_SEP + "PCA_12.csv"
CELL_LINES_14 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + DATA_FOLDER + SYSTEM_SEP + "PCA_14.csv"
CELL_LINES_17 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + DATA_FOLDER + SYSTEM_SEP + "PCA_17.csv"
CELL_LINES_22 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + DATA_FOLDER + SYSTEM_SEP + "PCA_22.csv"