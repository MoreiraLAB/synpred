"""
Script to deploy tensorflow for Gridsearch
conda activate tf
tensorflow version 2
"""

import pandas as pd
from sklearn import preprocessing
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, \
                            recall_score, precision_score, f1_score
from DEC_variables import RANDOM_STATE, DROPPABLE_COLUMNS, TARGET_CLASS_COLUMN, \
							CSV_SEP, TRAIN_DATASET_PCA, TEST_DATASET_PCA, \
                            TSNE_PLOT_FILE, PARAGRAPH_SEP, TSNE_TABLE_FILE, \
                            PCA_PLOT_FILE, PCA_TABLE_FILE
import sys
import random
import numpy as np
import ast
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def prepare_dataset(input_train, input_test, drop_columns = DROPPABLE_COLUMNS, \
                    target_column = TARGET_CLASS_COLUMN, subset_size = 0):
    
    if subset_size != 0:
        train_dataset = pd.read_csv(input_train, sep = CSV_SEP, header = 0, nrows = subset_size)
        test_dataset = pd.read_csv(input_test, sep = CSV_SEP, header = 0, nrows = subset_size)
    else:
        train_dataset = pd.read_csv(input_train, sep = CSV_SEP, header = 0)
        test_dataset = pd.read_csv(input_test, sep = CSV_SEP, header = 0)
    train_class = train_dataset[target_column]
    train_features = train_dataset.drop(drop_columns, axis = 1)
    test_class = test_dataset[target_column]
    test_features = test_dataset.drop(drop_columns, axis = 1)

    return {"train_features": train_features, "train_class": train_class, \
            "test_features": test_features, "test_class": test_class}

datasets_dictionary = {"PCA": [TRAIN_DATASET_PCA, TEST_DATASET_PCA]}
proper_dictionary = prepare_dataset(datasets_dictionary["PCA"][0], datasets_dictionary["PCA"][1])
pca = PCA(n_components = 2)
pca.fit(pd.concat([proper_dictionary["train_features"], proper_dictionary["test_features"]], axis = 0))

pca_scaler = StandardScaler()
transformed_data = pca_scaler.fit_transform(pca.transform(pd.concat([proper_dictionary["train_features"], proper_dictionary["test_features"]], axis = 0)))

pca_plot_table = pd.DataFrame()
pca_plot_table["Principal component one"] = transformed_data[:,0]
pca_plot_table["Principal component two"] = transformed_data[:,1]
pca_plot_table["Class"] = list(proper_dictionary["train_class"].values) + list(proper_dictionary["test_class"].values)
pca_plot_table["Class"].loc[(pca_plot_table["Class"] == 1)] = "Synergistic"
pca_plot_table["Class"].loc[(pca_plot_table["Class"] == 0)] = "Non-synergistic"
pca_plot_table.to_csv(PCA_TABLE_FILE, index = False)

pca_plot = sns.scatterplot(
    x  ="Principal component one", y = "Principal component two",
    hue = "Class",
    palette = sns.color_palette("hls", 2),
    data = pca_plot_table,
    legend = "full",
    alpha  = 0.6
)
fig = pca_plot.get_figure()
fig.savefig(PCA_PLOT_FILE)

tsne = TSNE(n_components = 2, verbose = 2, perplexity = 50, n_iter = 1000)
tsne_scaler = StandardScaler()
tsne_results = tsne_scaler.fit_transform(tsne.fit_transform(pd.concat([proper_dictionary["train_features"], proper_dictionary["test_features"]], axis = 0)))

tsne_plot_table = pd.DataFrame()
tsne_plot_table["T-sne component one"] = tsne_results[:,0]
tsne_plot_table["T-sne component two"] = tsne_results[:,1]
tsne_plot_table["Class"] = list(proper_dictionary["train_class"].values) + list(proper_dictionary["test_class"].values)
tsne_plot_table["Class"].loc[(tsne_plot_table["Class"] == 1)] = "Synergistic"
tsne_plot_table["Class"].loc[(tsne_plot_table["Class"] == 0)] = "Non-synergistic"
tsne_plot_table.to_csv(TSNE_TABLE_FILE, index = False)

tsne_plot = sns.scatterplot(
    x  ="T-sne component one", y = "T-sne component two",
    hue = "Class",
    palette = sns.color_palette("hls", 2),
    data = tsne_plot_table,
    legend = "full",
    alpha  = 0.6
)
fig = tsne_plot.get_figure()
fig.savefig(TSNE_PLOT_FILE)