import itertools
import pickle
import numpy as np
import pandas as pd
import random
import collections
import math as mt
from joblib import Parallel, delayed
import timeit
from condtional_prob import causal_prob, conditional_prob, get_probabiity
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)


def get_baseline(X, model):
    fx = 0
    n_features = X.shape[1]
    X = np.reshape(X, (len(X), 1, n_features))
    for i in X:
        fx += model.predict(i)
    return fx / len(X)


# Returns value from using function for different versions
def get_value(version, permutation, X, x, unique_count, causal_struct, model, N, is_classification, xi):
    # intializing returns
    absolute_diff, f1, f2 = 0, 0, 0
    xi_index = permutation.index(xi)
    indices = permutation[:xi_index + 1]
    indices_baseline = permutation[xi_index + 1:]
    x_hat = np.zeros(N)
    x_hat_2 = np.zeros(N)
    len_X = len(X)
    for j in indices:
        x_hat[j] = x[j]
        x_hat_2[j] = x[j]
    if version == '2' or version == '3' or version == '4':
        proba1, proba2 = 0, 0
        baseline_check_1, baseline_check_2 = [], []
        f1, f2 = 0, 0
        indices_baseline_2 = indices_baseline[:]
        for index, i in enumerate(unique_count):
            # print(index, '/', len(unique_count))
            X = np.asarray(i)
            for j in indices_baseline:
                x_hat[j] = X[j]
                x_hat_2[j] = X[j]

            # No repetition
            # Eg if baseline_indices is null, it'll only run once as x_hat will stay the same over each iteration
            if x_hat.tolist() not in baseline_check_1:
                baseline_check_1.append(x_hat.tolist())
                if version == '2':
                    prob_x_hat = get_probabiity(unique_count, x_hat, indices_baseline, len_X)
                elif version == '3':
                    prob_x_hat = conditional_prob(unique_count, x_hat, indices, indices_baseline, len_X)
                else:
                    prob_x_hat = causal_prob(unique_count, x_hat, indices, indices_baseline, causal_struct, len_X)
                proba1 += prob_x_hat
                x_hat = np.reshape(x_hat, (1, N))
                f1 = f1 + (model.predict_proba(x_hat)[0][1] * prob_x_hat if is_classification else model.predict(
                    x_hat) * prob_x_hat)

            # xi index will be given to baseline for f2
            x_hat_2[xi] = X[xi]
            if xi not in indices_baseline_2:
                indices_baseline_2.append(xi)

            # No repetition
            indices_2 = indices[:]
            indices_2.remove(xi)
            if x_hat_2.tolist() not in baseline_check_2:
                baseline_check_2.append(x_hat_2.tolist())
                if version == '2':
                    prob_x_hat_2 = get_probabiity(unique_count, x_hat_2, indices_baseline_2, len_X)
                elif version == '3':
                    prob_x_hat_2 = conditional_prob(unique_count, x_hat_2, indices_2, indices_baseline_2, len_X)
                else:
                    prob_x_hat_2 = causal_prob(unique_count, x_hat_2, indices_2, indices_baseline_2, causal_struct,
                                               len_X)
                proba2 += prob_x_hat_2
                x_hat_2 = np.reshape(x_hat_2, (1, N))
                f2 = f2 + (model.predict_proba(x_hat_2)[0][1] * prob_x_hat_2 if is_classification else model.predict(
                    x_hat_2) * prob_x_hat_2)
            x_hat = np.squeeze(x_hat)
            x_hat_2 = np.squeeze(x_hat_2)
        absolute_diff = abs(f1 - f2)
    elif version == '1':
        f1, f2 = 0, 0
        for i in range(len(X)):
            # print(i)
            for j in indices_baseline:
                x_hat[j] = X[i][j]
                x_hat_2[j] = X[i][j]
            x_hat = np.reshape(x_hat, (1, N))
            f1 += model.predict_proba(x_hat)[0][1] if is_classification else model.predict(x_hat)
            x_hat_2[xi] = X[i][xi]
            x_hat_2 = np.reshape(x_hat_2, (1, N))
            f2 += model.predict_proba(x_hat_2)[0][1] if is_classification else model.predict(x_hat_2)
            x_hat = np.squeeze(x_hat)
            x_hat_2 = np.squeeze(x_hat_2)
        absolute_diff = abs(f1 - f2) / len_X
        f1 = f1 / len_X
        f2 = f2 / len_X
    return absolute_diff, f1, f2


def approximate_shapley(version, xi, N, X, x, m, model, unique_count, causal_struct, is_classification,
                        global_shap=False):
    # R = list(itertools.permutations(range(N)))
    # random.shuffle(R)
    score = 0
    count_negative = 0
    vf1, vf2 = 0, 0
    # res = [random.randrange(1, m, 1) for i in range(1000)]
    for index, i in enumerate(itertools.permutations(range(N))):
        # if index in res:
            # print(index, '/', m)
        abs_diff, f1, f2 = get_value(version, list(i), X, x, unique_count, causal_struct, model, N,
                                     is_classification, xi)
        vf1 += f1
        vf2 += f2
        score += abs_diff
        if not global_shap:
            if vf2 > vf1:
                count_negative -= 1
            else:
                count_negative += 1
    if count_negative < 0 and not global_shap:
        score = -1 * score
    return score / m


def shapley_main(version, df_X, model, causal_struct, local_shap=0, global_shap=True, is_classification=False):
    sigma_phi = 0
    feature_names = df_X.columns
    X = df_X.values
    n_features = len(feature_names)
    unique_count = collections.Counter(map(tuple, X))
    ##### f(x) with baseline
    f_o = get_baseline(X, model)
    # f_o = 0.0249
    baseline = np.mean(X, axis=0)
    # To convert baseline (mean) to discrete
    if is_classification:
        baseline = [1 if i > 0.5 else 0 for i in baseline]
    print("Baseline f(x): ", f_o)
    feature_score_list = []
    if global_shap:
        for feature in range(n_features):
            global_shap_score = Parallel(n_jobs=-1)(
                delayed(approximate_shapley)(version, feature, n_features, X, x, mt.factorial(n_features), model,
                                             unique_count, causal_struct, is_classification, global_shap) for i, x in
                enumerate(X))
            global_shap_score = sum(global_shap_score)
            global_shap_score = global_shap_score / len(X)
            print("x", str(feature + 1), ": ", global_shap_score)
    else:
        x = X[local_shap]
        for feature in range(n_features):
            local_shap_score = approximate_shapley(version, feature, n_features, X, x, mt.factorial(n_features), model,
                                                   unique_count, causal_struct,
                                                   is_classification)
            print("x", str(feature + 1), ": ", local_shap_score)
            feature_score_list.append(local_shap_score)
            sigma_phi += local_shap_score
        x = np.reshape(x, (1, n_features))
        print("local f(x): ", model.predict(x))
        # Making use of Sigma_phi = f(x) + f_o
        # Where f_o = E(f(X))
        print("Sigma_phi + E(fX): ", sigma_phi + f_o)
    sns.barplot(y=feature_names, x=feature_score_list)
    plt.show()
