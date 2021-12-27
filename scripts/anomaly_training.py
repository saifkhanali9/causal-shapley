import os
from pathlib import Path
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
import tensorflow
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.auto_encoder_torch import AutoEncoder as encoder_torch

from joblib import dump, load
from tensorflow.keras.losses import binary_crossentropy
import torch.nn as nn

dataset_name = 'census2'
model_name = '/encoder_torch'
model_file = "../output/model/"
model_path = model_file + dataset_name + model_name
os.makedirs(model_file + dataset_name, exist_ok=True)

def load_save_model(model, x):
    my_file = Path(model_path + ".joblib")
    if my_file.is_file():
        model = load(model_path + '.joblib')
    else:
        model.fit(x)
        dump(model, model_path + ".joblib")
    return model


def ann_pyod(x, x_test, is_torch):
    if is_torch:
        clf = AutoEncoder(hidden_neurons=[x.shape[1], 8, 4, 2, 4, 8, x.shape[1]], epochs=300, loss=binary_crossentropy)
    else:
        clf = encoder_torch(hidden_neurons=[x.shape[1], 8, 4, 2, 4, 8, x.shape[1]], epochs=300, loss_fn=nn.BCEWithLogitsLoss)
    clf = load_save_model(clf, x)

    # get the prediction on the test data
    y_test_pred = clf.predict(x)  # outlier labels (0 or 1)
    unique, counts = np.unique(y_test_pred, return_counts=True)
    train = dict(zip(unique, counts))
    print("Train acc: ", round((train[0] / len(y_test_pred)), 4))
    y_test_scores = clf.predict(x_test)  # outlier scores
    unique, counts = np.unique(y_test_scores, return_counts=True)
    test = dict(zip(unique, counts))
    # print("Test acc: ", test[0] / len(y_test_scores))
    print(y_test_pred)
    print(y_test_scores)



data_path = '../output/dataset/'
x_train = pd.read_csv(data_path + dataset_name + '/x_train.csv')
x_test = pd.read_csv(data_path + dataset_name + '/x_test.csv')
frames = [x_train, x_test]
x_train = pd.concat(frames)

# x_test = np.array([[43, 57, 17, 1, 44, 61, 19, 50]])
# # x_test = x_test.drop(['capital_loss', 'capital_gain', 'race', 'marital_status', 'sex', 'workclass', 'relationship'],
# #                      axis=1)
# y_test = pd.read_csv(data_path + '/y_train.csv')
ann_pyod(x_train, x_test, is_torch=True)
