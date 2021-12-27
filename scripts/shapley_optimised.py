import pickle

import time
import numpy as np
import pandas as pd
import math
import torch
import seaborn as sns
from joblib import load, Parallel, delayed
from matplotlib import pyplot as plt

def calc_permutations(S, p):
    return (math.factorial(len(S)) * math.factorial(p - len(S) - 1)) / math.factorial(p)


def show_values(axs, orient="h", space=0):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def predict(x, model, is_classification):
    if False:
        if len(x.shape) > 1:
            return model.predict_proba(x)[:, 1]
        return model.predict_proba(x.reshape(1, -1))[0][1]
    else:
        if len(x.shape) > 1:
            return torch.from_numpy(model.predict(x)).type(torch.FloatTensor)
        return model.predict(x.reshape(1, -1))[0]


def get_baseline(X, model, is_classification=True):
    fx = torch.mean(predict(X, model, is_classification))
    return fx


def powerset(features):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    listrep = list(features)
    n = len(listrep)
    return [[listrep[k] for k in range(n) if i & 1 << k] for i in range(2 ** n)]


def baseline(X, x, features_baseline, model, is_classification):
    temp_row = torch.zeros(X.shape)
    temp_row[:] = x
    temp_row[:, features_baseline] = X[:, features_baseline]
    f1 = torch.mean(predict(temp_row, model, is_classification))
    return f1


def multithreaded_powerset(X, x, s, s_index, feature_index, features_list, model, is_classification):
    # for count_power, s in enumerate(S):
    print(s_index, '/', 2 ** len(x))
    s_baseline = list(set(s).symmetric_difference(features_list))
    s_union_j_baseline = s_baseline[:]
    s_union_j_baseline.remove(feature_index)
    v_u_j = baseline(X, x, s_union_j_baseline, model, is_classification)
    v = baseline(X, x, s_baseline, model, is_classification)
    # phi_i += calc_permutations(S=s, p=X.shape[1]) * (v_u_j - v)
    return calc_permutations(S=s, p=X.shape[1]) * (v_u_j - v)


def shap_optimized(X, x, feature_names, model, is_classification):
    features_list = list(range(len(feature_names)))
    feature_scores = []
    # Loop for phi_i
    total_time = 0
    for i in features_list:
        start_time = time.time()
        features_list_copy = features_list[:]
        del features_list_copy[i]
        S = powerset(features_list_copy)
        phi_i = 0
        for count_power, s in enumerate(S):
            print(count_power, '/', 2 ** len(features_list))
            s_baseline = list(set(s).symmetric_difference(features_list))
            s_union_j_baseline = s_baseline[:]
            s_union_j_baseline.remove(i)
            v_u_j = baseline(X, x, s_union_j_baseline, model, is_classification)
            v = baseline(X, x, s_baseline, model, is_classification)
            phi_i += calc_permutations(S=s, p=X.shape[1]) * (v_u_j - v)
            # break
        phi_i = Parallel(n_jobs=-1)(
            delayed(multithreaded_powerset)(X, x, s, index, i, features_list, model, is_classification) for index, s in
            enumerate(S))
        # print(phi_i)
        phi_i = phi_i.item()
        diff_time = round(time.time() - start_time, 3)
        total_time += diff_time
        print("Feature", i, " | ", feature_names[i], ": \t\t", round(phi_i, 4), '\t Time taken: ', diff_time, 'sec')
        feature_scores.append(phi_i)
    print('\nTotal time: ', round(total_time, 4), ' sec\n\nBaseline: ', baseline_V, '\nSigma_phi: ',
          sum(feature_scores))
    # Making use of Sigma_phi = f(x) + f_o
    # Where f_o = E(f(X))
    print("\nlocal f(x):\t\t\t", predict(x, model, is_classification), "\nSigma_phi + E(fX):\t",
          round(sum(feature_scores) + baseline_V, 7))
    return feature_scores


file_name = 'census2'
is_classification = True
local_index = 15
file_path = '../output/dataset/' + file_name + '/x_train.csv'
# file_path = '../output/dataset/' + file_name + '.csv'
# model_path = '../output/model/' + file_name + '.sav'
# model_path = '../output/model/' + file_name + '/xgb_clf.pkl'
# model = pickle.load(open(model_path, 'rb'))
df = pd.read_csv(file_path)
feature_names = df.columns.tolist()
X = df.to_numpy()
# X = X[:, :-1]
# X = X[:16, :]
print(X.shape)
X_torch = torch.from_numpy(X).type(torch.FloatTensor)
model = load('../output/model/census2/encoder_torch.joblib')
baseline_V = get_baseline(X, model, is_classification)
# temp_numpy = np.array([58, 57, 8, 7, 41, 9, 29, 76, 39, 0, 0, 136, 73])
temp_numpy = np.loadtxt('../notebooks/age_hpw_anomaly.txt')[1]
x_torch = torch.from_numpy(temp_numpy).type(torch.FloatTensor)
feature_score_list = shap_optimized(X_torch, x_torch, feature_names, model, is_classification)
# feature_score_list, feature_names = zip(*sorted(zip(feature_score_list, feature_names)))
sns.barplot(y=feature_names, x=feature_score_list)
# show_values(p)
plt.show()
