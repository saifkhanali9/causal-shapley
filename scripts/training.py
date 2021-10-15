from pathlib import Path
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle

from sklearn.linear_model import LinearRegression, LogisticRegression
import os


def train(model_type='regression', file_name='synthetic1', save_model=False):
    file = '../output/dataset/' + file_name + '.csv'
    df = pd.read_csv(file)
    X = df.iloc[:, :4]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    if model_type == 'regression':
        model = LinearRegression().fit(X_train, y_train)
    else:
        model = LogisticRegression(random_state=0).fit(X_train, y_train)
        # model = svm.SVC(kernel='rbf')
        # model.fit(X_train, y_train)
    print("Test score: ", model.score(X_test, y_test))
    if save_model:
        try:
            os.makedirs('../output/dataset/' + file_name)
        except FileExistsError:
            pass
        try:
            os.makedirs('../output/model')
        except FileExistsError:
            pass
        model_file = '../output/model/' + file_name + '.sav'
        pickle.dump(model, open(model_file, 'wb'))
        X_train.to_csv(Path('../output/dataset/') / file_name / 'X_train.csv', index=False)
        X_test.to_csv(Path('../output/dataset/') / file_name / 'X_test.csv', index=False)
        y_train.to_csv(Path('../output/dataset/') / file_name / 'y_train.csv', index=False)
        y_test.to_csv(Path('../output/dataset/') / file_name / 'y_test.csv', index=False)


def model_test(folder_name='synthetic1'):
    X_test = pd.read_csv('../output/dataset/' + folder_name + '/X_test.csv')
    y_test = pd.read_csv('../output/dataset/' + folder_name + '/y_test.csv')
    loaded_model = pickle.load(open('../output/model/' + folder_name + '.sav', 'rb'))
    result = loaded_model.score(X_test, y_test)
    print("Testing: ", result)


train(model_type='classification',file_name='synthetic_discrete_2', save_model=True)
model_test(folder_name='synthetic_discrete_2')
