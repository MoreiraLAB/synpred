"""
List all usable variables
"""

"""
Folder paths
"""
HOME = "C:/users/marti/OneDrive/Documents/GitHub/synpred"
SYSTEM_SEP = "/"
SUPPORT_FOLDER = "support"
RESOURCES_FOLDER = "resources"
STANDALONE_MODELS = "standalone_models"
TENSORFLOW_MODEL_PATH_1 = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "1_after_grid_save.h5"
TENSORFLOW_MODEL_PATH_2 = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "3_after_grid_save.h5"
TENSORFLOW_MODEL_PATH_3 = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "5_after_grid_save.h5"
TENSORFLOW_MODEL_PATH_4 = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "7_after_grid_save.h5"
TENSORFLOW_MODEL_PATH_ENSEMBLE = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "final_ensemble.h5"
XGB_MODEL_PATH = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "PCA_XGB.pkl"
SVM_MODEL_PATH = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "PCA_SVM.pkl"
KNN_MODEL_PATH = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "PCA_KNN.pkl"
ETC_MODEL_PATH = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "PCA_ETC.pkl"
SGD_MODEL_PATH = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "PCA_SGD.pkl"
MLP_MODEL_PATH = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "PCA_MLP.pkl"
RF_MODEL_PATH = HOME + SYSTEM_SEP + STANDALONE_MODELS + SYSTEM_SEP + "PCA_RF.pkl"
TENSORFLOW_CHECKPOINT = HOME + SYSTEM_SEP + SUPPORT_FOLDER
SEP = ","
INTERMEDIATE_SEP = "_"
TXT_TERMINATION = ".txt"
CSV_TERMINATION = ".csv"
PROCESSED_TERMINATION = INTERMEDIATE_SEP + "processed" + CSV_TERMINATION
FEATURES_TERMINATION = INTERMEDIATE_SEP + "features" + CSV_TERMINATION
PREDICTION_COL_NAME = "Ensemble prediction"
PREDICTION_COL_NAME_DL1 = "DL1 prediction"
PREDICTION_COL_NAME_DL2 = "DL2 prediction"
PREDICTION_COL_NAME_DL3 = "DL3 prediction"
PREDICTION_COL_NAME_DL4 = "DL4 prediction"
PREDICTION_COL_NAME_XGB = "XGBoost prediction"
PREDICTION_COL_NAME_ETC = "ETC prediction"
PREDICTION_COL_NAME_RF = "RF prediction"
PREDICTION_COL_NAME_KNN = "KNN prediction"
PREDICTION_COL_NAME_MLP = "MLP prediction"
PREDICTION_COL_NAME_SVM = "SVM prediction"
PREDICTION_COL_NAME_SGD = "SGD prediction"

"""
Cell lines files
"""
CELL_LINES_FILE = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + "almanac_cell_lines_library.csv"
CELL_LINES_ID = "NCI-60"
CELL_LINES_TYPE = "TISSUE"
CELL_LINES_COLUMN = "cells"
CELL_LINES_5 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + "5_CCLE.csv"
CELL_LINES_6 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + "6_CCLE.csv"
CELL_LINES_7 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + "7_CCLE.csv"
CELL_LINES_12 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + "12_CCLE.csv"
CELL_LINES_14 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + "14_CCLE.csv"
CELL_LINES_17 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + "17_CCLE.csv"
CELL_LINES_22 = HOME + SYSTEM_SEP + RESOURCES_FOLDER + SYSTEM_SEP + "22_CCLE.csv"
