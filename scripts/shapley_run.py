import json
import timeit
import pickle

import numpy as np
import pandas as pd

from causal_shapley import shapley_main


class SyntheticModel():
    def __init__(self):
        pass

    def predict(self, x):
        return x[0][0]

    def predict_proba(self, x):
        return np.array([[1-x[0][0], x[0][0]]])


# data_path = 'census/x_test'
data_path = 'synthetic_cont_1'
# data_path = 'temp'
# model_path = 'census/xgb_clf.pkl'
model_path = 'synthetic_cont_1.sav'
df = pd.read_csv('../output/dataset/' + data_path + '.csv')
X = df.iloc[:, :len(df.columns[:-1])]
# X = df
# X = X[:, :X.shape[1] - 1]
model = pickle.load(open('../output/model/' + model_path, 'rb'))
# model = SyntheticModel()
causal_struct = None
try:
    causal_struct = json.load(open('../output/dataset/' + data_path + '/causal_struct.json', 'rb'))
except FileNotFoundError:
    pass
start = timeit.default_timer()
shapley_main(version='1', df_X=X, model=model, causal_struct=causal_struct, local_shap=15, is_classification=False,
             global_shap=False)
stop = timeit.default_timer()
print('Time: ', stop - start)
