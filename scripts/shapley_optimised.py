import pickle

import time
import numpy as np
import pandas as pd
import math


class SyntheticModel():
    def __init__(self):
        pass

    def predict(self, x):
        return x[0]


def calc_permutations(S, p):
    return (math.factorial(len(S)) * math.factorial(p - len(S) - 1)) / math.factorial(p)


def predict(x, model, is_classification):
    if is_classification:
        if len(x.shape) > 1:
            return model.predict_proba(x)[:, 1]
        return model.predict_proba(x.reshape(1, -1))[0]
    else:
        if len(x.shape) > 1:
            return model.predict(x)
        return model.predict(x.reshape(1, -1))[0]


def create_s_u_j(s, j):
    if len(s) == 0:
        return [j]
    for i, val in enumerate(s):
        if j < s[i] and i == (len(s) - 1):
            s.insert(i, j)
            break
        if j < s[i] and i == 0:
            s.insert(i, j)
            break
        if j > s[i] and i == (len(s) - 1):
            s.append(j)
            break
        elif j > s[i] and j < s[i + 1]:
            s.insert(i + 1, j)
            break
    return s


def get_baseline(X, model, is_classification=True):
    fx = np.mean(predict(X, model, is_classification))
    return fx


def powerset(features):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    listrep = list(features)
    n = len(listrep)
    return [[listrep[k] for k in range(n) if i & 1 << k] for i in range(2 ** n)]


def baseline(X, x, model, features_baseline, is_classification):
    temp_row = np.zeros(X.shape)
    temp_row[:] = x
    temp_row[:, features_baseline] = X[:, features_baseline]
    f1 = np.mean(predict(temp_row, model, is_classification))
    return f1


def shap_optimized(x, model, is_classification):
    features_list = list(range(len(x)))
    # Loop for phi_i
    phi = 0
    for i in features_list:
        time1 = time.time()
        features_list_copy = features_list[:]
        del features_list_copy[i]
        S = powerset(features_list_copy)
        phi_i = 0
        for s in S:
            s_baseline = list(set(s).symmetric_difference(features_list))
            s_union_j_baseline = s_baseline[:]
            s_union_j_baseline.remove(i)
            v_u_j = baseline(X, x, s_union_j_baseline, model, is_classification)
            v = baseline(X, x, s_baseline, model, is_classification)
            phi_i += calc_permutations(S=s, p=X.shape[1]) * (v_u_j - v)
        print('Feature ', i + 1, ": ", phi_i, '\tTime taken: ', time.time() - time1, '\n')
        phi += phi_i
    print("local f(x): ", predict(x, model, is_classification)[1], '\nBaseline: ', baseline_V, '\nSigma_phi: ', phi)
    print("Sigma_phi + E(fX): ", phi + baseline_V)


file_name = 'synthetic_discrete_3'
is_classification = True
file_path = '../output/dataset/' + file_name + '.csv'
# file_path = '../output/dataset/' + file_name + '.csv'
model_path = '../output/model/' + file_name + '.sav'
# model_path = '../output/model/' + file_name + '/xgb_clf.pkl'
model = pickle.load(open(model_path, 'rb'))
X = pd.read_csv(file_path).to_numpy()
X = X[:10000, :-1]
# X = X[:, :]
print(X.shape)
baseline_V = get_baseline(X, model, is_classification)
shap_optimized(X[15], model, is_classification)
