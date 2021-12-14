import json
import timeit
import pickle
import pandas as pd

from causal_shapley import shapley_main

# data_path = 'census/x_test'
data_path = 'synthetic_discrete_3'
# model_path = 'census/xgb_clf.pkl'
model_path = 'synthetic_discrete_3.sav'
df = pd.read_csv('../output/dataset/' + data_path + '.csv')
X = df.iloc[:, :len(df.columns[:-1])]
# X = df
# X = X[:, :X.shape[1] - 1]
model = pickle.load(open('../output/model/' + model_path, 'rb'))
causal_struct = None
try:
    causal_struct = json.load(open('../output/dataset/' + data_path + '/causal_struct.json', 'rb'))
except FileNotFoundError:
    pass
start = timeit.default_timer()
shapley_main(version='2', df_X=X, model=model, causal_struct=causal_struct, local_shap=14, is_classification=True,
             global_shap=False)
stop = timeit.default_timer()
print('Time: ', stop - start)