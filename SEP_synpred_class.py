"""
Script to deploy tensorflow for Gridsearch
conda activate tf
tensorflow version 2
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.losses import BinaryCrossentropy
import pandas as pd
from sklearn import preprocessing
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, \
                            recall_score, precision_score, f1_score
import sys
import random
import numpy as np
import ast

np.random.seed(1)
random.seed(1)
tf.compat.v1.set_random_seed(1)
c_type = 'all'

class neural_network_class:

    def __init__(self, input_architecture,  \
                        input_features, \
                        activation_function = "relu", \
                        add_dropout = True, \
                        dropout_rate = 0.5):
        self.model = Sequential()
        self.model.add(Dense(input_architecture[0], input_dim = input_features, \
                                    activation = activation_function, \
                                    kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                    bias_regularizer = regularizers.l2(1e-4), \
                                    activity_regularizer = regularizers.l2(1e-5)))
        for hidden_layer in input_architecture[1:]:
            if add_dropout == True:
                self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(hidden_layer, activation = activation_function, \
                                    kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                    bias_regularizer = regularizers.l2(1e-4), \
                                    activity_regularizer = regularizers.l2(1e-5)))
            
        self.model.add(Dense(1, activation='sigmoid'))

def make_ds(file='train_biclassification_full_agreement_norm_encoder.csv', method='ZIP', scaler=None, c_type='all', sample_fraction = 1.0):
    df = pd.read_csv(file)
    df = df.sample(frac = sample_fraction)

    cell_types = {
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

    if c_type != 'all':
        if type(c_type) is list:
            exclusion = cell_types[c_type[0]]
            for i in c_type[1:]:
                exclusion += cell_types[i]
        else:
            exclusion = cell_types[c_type]

    if c_type != 'all':
        df = df.loc[df['cell'].isin(exclusion)]

    names = df[['cell', 'drug1', 'drug2']]
    features = df

    droper = ['cell', 'drug1', 'drug2', 'ZIP', 'Bliss', 'HSA','Full-agreement']

    for column in droper:
        if column != method:
            features = features.drop([column], axis = 1)
        else:
            continue

    target = features.pop(method)
    features = features #scaler.fit_transform(features)

    #target = target.replace({0: 0, 2: 1, 3: 2})

    return names, features, target

def model_training(scaler, ds, method, input_model, c_type='all', outp=None, save_model = True):

    names, features, target = make_ds(file=ds, method=method, scaler = scaler, c_type=c_type, sample_fraction = 0.1)
    history = input_model.model.fit(x = features, y = target, epochs = 250, validation_split=0.10)
    if save_model == True:
        input_model.model.save('./saved_model/{}_{}'.format(outp, method))

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_csv('train_log/training_metrics_{}_{}.csv'.format(outp, method))
    predicted_class = input_model.model.predict(features)
    return input_model, scaler, target, predicted_class

def model_evaluation(input_class, input_predictions, \
                        subset_type = "test", verbose = False):
    
    output_file_name = "evaluation_summary/" + subset_type + ".csv"
    accuracy = accuracy_score(input_class, input_predictions)
    precision = precision_score(input_class, input_predictions)
    recall = recall_score(input_class, input_predictions)
    auc = roc_auc_score(input_class, input_predictions)
    f1_value = f1_score(input_class, input_predictions)
    if verbose == True:
        print("Accuracy:", round(accuracy, 2), "\n",
               "Precision:" , round(precision, 2), "\n",
            "Recall:", round(recall, 2), "\n",
            "AUC:", round(auc, 2), "\n",
            "F1-score:", round(f1_value, 2), "\n")
    with open(output_file_name, "w") as output_file:
        output_file.write("Metric,Value\n")
        output_file.write("Accuracy," + str(accuracy) + "\n")
        output_file.write("Precision," + str(precision) + "\n")
        output_file.write("Recall," + str(recall) + "\n")
        output_file.write("AUC," + str(auc) + "\n")
        output_file.write("F1-score," + str(f1_value) + "\n")

input_mode = sys.argv[3]
method = "ZIP"
train = "./datasets/train_" + input_mode + ".csv"
test = "./datasets/test_" + input_mode + ".csv"
outp = sys.argv[-1]
standard_scaler = preprocessing.StandardScaler()
input_architecture = [int(x) for x in ast.literal_eval(sys.argv[1])]

if input_mode.split("_")[0] == "PCA":
    input_features = 1347
elif input_mode.split("_")[0] == "autoencoder":
    input_features = 4229

raw_model =  neural_network_class(input_architecture, input_features, dropout_rate = float(sys.argv[2]))
optimizer = tf.keras.optimizers.Adam(0.0001)
raw_model.model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
classification_model, new_scaler, train_class, train_predictions = model_training(standard_scaler, train, \
                                                    method, raw_model, c_type=c_type, outp=outp, save_model = False)

names, test, target = make_ds(file = test, method=method, scaler=new_scaler, c_type=c_type)
test_predictions = [int(np.round(x)) for x in classification_model.model.predict(test)]
train_predictions = [int(np.round(x)) for x in train_predictions]
model_evaluation(target, test_predictions, subset_type = outp + "_test")
model_evaluation(train_class, train_predictions, subset_type = outp + "_train")
sys.exit()

names['target'] = list(target)
names['prediction'] = list(test_predictions)

classification_model.model.summary()

names.to_csv('predictions_{}_{}.csv'.format(outp, method))