import collections
import pickle

import numpy as np
import pandas as pd
import math
from itertools import chain, combinations


class SyntheticModel():
    def __init__(self):
        pass

    def predict(self, x):
        return x[0]


def calc_permutations(S, p):
    return (math.factorial(len(S)) * math.factorial(p - len(S) - 1)) / math.factorial(p)


def create_s_u_j(s, j):
    if len(s) == 0:
        return [j]
    # if j < s[0]:
    #
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


def get_baseline(X, model):
    fx = 0
    n_features = X.shape[1]
    X = np.reshape(X, (len(X), 1, n_features))
    for i in X:
        fx += model.predict(i)[0]
    return fx / len(X)


# Probability is taken over indices of baseline only
def get_probability(unique_count, x_hat, indices_baseline, n):
    # if len(indices_baseline) > 0:
    count = 0
    for i in unique_count:
        # check = True
        key = np.asarray(i)
        # for j in indices_baseline:
        if np.array_equal(key[indices_baseline], x_hat[indices_baseline]):
            # if check:
            count += unique_count[i]
    return count / n
    # else:
    #     return 1


def powerset(features):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    # s = list(iterable)
    # return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    listrep = list(features)
    n = len(listrep)
    return [[listrep[k] for k in range(n) if i & 1 << k] for i in range(2 ** n)]


baseline_dict_prob = {}
baseline_dict_value = {}
value_function = {}


def baseline(x, features_baseline, model):
    # feature_key = " ".join(str(f) for f in features_baseline)
    v = 0
    temp_row = np.zeros(len(x))
    temp_row[:] = x
    temp_row[features_baseline] = np.zeros(len(features_baseline))
    temp_v = model.predict(temp_row.reshape(1, -1))[0]
    v += temp_v

    # value_function[feature_key] = v
    return v


def shap_optimized(X, x, model):
    features_list = list(range(len(x)))
    phi_list = np.zeros(len(x))
    unique_count = collections.Counter(map(tuple, X))
    # Loop for phi_i
    phi = 0
    for i in features_list:
        features_list_copy = features_list[:]
        # S1 = powerset(features_list)
        del features_list_copy[i]
        S = powerset(features_list_copy)
        phi_i = 0
        count_neg = 0
        for s in S:
            s_baseline = list(set(s).symmetric_difference(features_list))
            s_union_j_baseline = s_baseline[:]
            s_union_j_baseline.remove(i)
            v_u_j = baseline(x, s_union_j_baseline, model)
            v = baseline(x, s_baseline, model)
            if v_u_j >= v:
                count_neg -= 1
            else:
                count_neg += 1
            phi_i += calc_permutations(S=s, p=X.shape[1]) * (v_u_j - v)
        phi_list[i] = phi_i
        print(i, ": ", phi_i)
        phi += phi_i
    # print("local f(x): ", model.predict(x.reshape(1, -1)), '\nBaseline: ', baseline_V, '\nSigma_phi: ', phi)
    # Making use of Sigma_phi = f(x) + f_o
    # Where f_o = E(f(X))
    # print("Sigma_phi + E(fX): ", phi + baseline_V)
    return phi_list


file_name = 'synthetic_cont_2'
file_path = '../output/dataset/' + file_name + '.csv'
model_path = '../output/model/' + file_name + '.sav'
model = pickle.load(open(model_path, 'rb'))
local_index = 15
X = pd.read_csv(file_path).to_numpy()
X = X[:1000, :-1]
tests_passed = 0
baseline_V = get_baseline(X, model)
# for i, row in enumerate(X):
#     explanations = np.round_(shap_optimized(row, model), 6)
#     XW = np.round_(np.multiply(row, model.coef_), 6)
#     if (np.array_equal(explanations, XW)):
#         tests_passed += 1
#     if (i+1) % 100 == 0:
#         print('Tests_passed: ', tests_passed, '/', len(X))
# print('Tests_passed: ', tests_passed, '/', len(X))
weights = np.zeros(len(X[local_index]))
weights[:] = model.coef_
weights[-1] = 0
print("x: ", X[local_index], '\nW: ', weights, '\nX*W', np.multiply(X[local_index], weights))
shap_optimized(X, X[local_index], model)
