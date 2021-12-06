import os
from pathlib import Path
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
import tensorflow
from pyod.models.auto_encoder import AutoEncoder
from joblib import dump, load

model_name = 'clf_autoencoder'
model_path = "../output/model/census/" + model_name

def load_save_model(model, x):
    my_file = Path(model_path + ".joblib")
    if my_file.is_file():
        model = load(model_path + '.joblib')
    else:
        model.fit(x)
        dump(model, model_path + ".joblib")
    return model


def ann_pyod(x, x_test):
    clf = AutoEncoder(hidden_neurons=[x.shape[1], 2, 2, x.shape[1]], epochs=100)
    clf = load_save_model(clf, x)

    # get the prediction on the test data
    y_test_pred = clf.predict(x)  # outlier labels (0 or 1)
    unique, counts = np.unique(y_test_pred, return_counts=True)
    train = dict(zip(unique, counts))
    print("Train acc: ", train[0] / len(y_test_pred))
    y_test_scores = clf.predict(x_test)  # outlier scores
    unique, counts = np.unique(y_test_scores, return_counts=True)
    test = dict(zip(unique, counts))
    # print("Test acc: ", test[0] / len(y_test_scores))
    print(y_test_pred)
    print(y_test_scores)

data_path = '../output/dataset/census/'
x_train = pd.read_csv(data_path + '/x_train.csv')
x_test = pd.read_csv(data_path + '/x_test.csv')
frames = [x_train, x_test]
x_train = pd.concat(frames)

x_test = np.array([[43, 57, 17, 1, 44, 61, 19, 50]])
# x_test = x_test.drop(['capital_loss', 'capital_gain', 'race', 'marital_status', 'sex', 'workclass', 'relationship'],
#                      axis=1)
y_test = pd.read_csv(data_path + '/y_train.csv')
ann_pyod(x_train, x_test)