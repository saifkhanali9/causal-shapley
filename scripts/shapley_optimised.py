import collections
import pickle

import numpy as np
import pandas as pd
import math
from itertools import chain, combinations


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
        if j > s[i] and i == (len(s) - 1):
            s.append(j)
            break
        elif j > s[i] and j > s[i + 1]:
            s.insert(i + 1, j)
            break
    return s


def get_baseline(X, model):
    fx = 0
    n_features = X.shape[1]
    X = np.reshape(X, (len(X), 1, n_features))
    for i in X:
        fx += model.predict(i)
    return fx / len(X)


# Probability is taken over indices of baseline only
def get_probability(unique_count, x_hat, indices_baseline, n):
    # if len(indices_baseline) > 0:
    count = 0
    for i in unique_count:
        check = True
        key = np.asarray(i)
        for j in indices_baseline:
            check = check and key[j] == x_hat[j]
        if check:
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


# baseline_dict_value_parent = {}


def baseline(X, x, features_in_use, unique_count, model):
    # feature_key = " ".join(str(f) for f in features_in_use)
    v = 0
    countt = 0
    # if feature_key in value_function.keys():
    #     return baseline_dict_prob[feature_key]
    if len(features_in_use) == len(x):
        return baseline_V
    else:
        baseline_check = []
        for row in X:
            # countt+= 1
            # print(countt)
            # temp_row = row[:]
            temp_row = np.zeros(len(x))
            temp_row[:] = x
            # for i in features_in_use:
            temp_row[features_in_use] = row[features_in_use]
            if features_in_use not in baseline_check:
                baseline_check.append(features_in_use)
                # if feature_key in baseline_dict_prob.keys():
                #     temp_p = baseline_dict_prob[feature_key]
                # else:
                temp_p = get_probability(unique_count, x, features_in_use, X.shape[0])
                #     # Above statement to not compute p again
                #     baseline_dict_prob[feature_key] = temp_p
                # if feature_key in baseline_dict_value.keys():
                #     temp_v = baseline_dict_value[feature_key]
                # else:
                temp_v = model.predict(temp_row.reshape(1, -1))
                    # baseline_dict_value[feature_key] = temp_v
                v += (temp_v * temp_p)
        # value_function[feature_key] = v
        return v


def shap_optimized(X, local_index, model):
    features_list = list(range(X.shape[1]))
    unique_count = collections.Counter(map(tuple, X))
    # Loop for phi_i
    phi = 0
    for i in features_list:
        features_list_copy = features_list[:]
        # S1 = powerset(features_list)
        del features_list_copy[i]
        S = powerset(features_list_copy)
        x = X[local_index]
        phi_i = 0
        count_neg = 0
        count_pos = 0
        for s in S:
            # print(s)
            # x = X[list(s)]
            s_union_j = s[:]
            s_baseline = list(set(s).symmetric_difference(features_list))
            s_union_j = create_s_u_j(s_union_j, i)
            s_union_j_baseline = list(set(s_union_j).symmetric_difference(features_list))
            v_u_j = baseline(X, x, s_union_j_baseline, unique_count, model)
            v = baseline(X, x, s_baseline, unique_count, model)
            if v_u_j >= v:
                count_pos += 1
            else:
                count_neg += 1
            # diff_v = v_u_j - v
            phi_i += calc_permutations(S=s, p=X.shape[1]) * (v_u_j - v)
        if count_pos < count_neg:
            phi_i *= -1
        print(i, ": ", phi_i)
        phi += phi_i
    print("local f(x): ", model.predict(x.reshape(1, -1)), '\nBaseline: ', baseline_V, '\nSigma_phi: ', phi)
    # Making use of Sigma_phi = f(x) + f_o
    # Where f_o = E(f(X))
    print("Sigma_phi + E(fX): ", phi + baseline_V)


file_name = 'synthetic_discrete_3'
file_path = '../output/dataset/' + file_name + '.csv'
model_path = '../output/model/' + file_name + '.sav'
model = pickle.load(open(model_path, 'rb'))
X = pd.read_csv(file_path).to_numpy()
X = X[:, :-1]
baseline_V = get_baseline(X, model)
shap_optimized(X, 14, model)
