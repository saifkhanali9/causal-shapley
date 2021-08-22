import itertools
import pickle
import numpy as np
import pandas as pd
import random
import collections, numpy
import math as mt
from condtional_prob import conditional

random.seed(42)


def get_baseline(X, model):
    fx = 0
    n_features = X.shape[1]
    X = np.reshape(X, (len(X), 1, n_features))
    for i in X:
        fx += model.predict(i)
    return fx / len(X)


def get_probabiity(unique_count, x_hat, indices_baseline, n):
    if len(indices_baseline) > 0:
        count = 0
        for i in unique_count:
            check = True
            key = np.asarray(i)
            for j in indices_baseline:
                check = check and key[j] == x_hat[j]
            if check:
                count += unique_count[i]
        return count / n
    else:
        return 1


def conditional_prob(unique_count, x_hat, indices, indices_baseline, n):
    numerator_indices = indices + indices_baseline
    numerator = get_probabiity(unique_count, x_hat, numerator_indices, n)
    denominator = get_probabiity(unique_count, x_hat, indices, n)
    return numerator / denominator


def get_expectation(X, x, indices, indices_baseline, baseline, unique_count, model, N, is_classification, xi,
                    version='4'):
    x_hat = np.zeros(N)
    x_hat_2 = np.zeros(N)
    len_X = len(X)
    for j in indices:
        x_hat[j] = x[j]
        x_hat_2[j] = x[j]
    if version == '3' or version == '4':
        proba1, proba2 = 0, 0
        baseline_check_1, baseline_check_2 = [], []
        f1, f2 = 0, 0
        indices_baseline_2 = indices_baseline[:]
        for i in unique_count:
            X = np.asarray(i)
            for j in indices_baseline:
                x_hat[j] = X[j]
                x_hat_2[j] = X[j]

            # No repetition
            # Eg if baseline_indices is null, it'll only run once as x_hat will stay the same over each iteration
            if x_hat.tolist() not in baseline_check_1:
                baseline_check_1.append(x_hat.tolist())
                prob_x_hat = get_probabiity(unique_count, x_hat, indices_baseline,
                                            len_X) if version == '3' else conditional_prob(unique_count, x_hat, indices,
                                                                                         indices_baseline, len_X)
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
                prob_x_hat_2 = get_probabiity(unique_count, x_hat_2, indices_baseline_2,
                                              len_X) if version == '3' else conditional_prob(unique_count, x_hat_2,
                                                                                           indices_2,
                                                                                           indices_baseline_2, len_X)
                proba2 += prob_x_hat_2
                x_hat_2 = np.reshape(x_hat_2, (1, N))
                f2 = f2 + (model.predict_proba(x_hat_2)[0][1] * prob_x_hat_2 if is_classification else model.predict(
                    x_hat_2) * prob_x_hat_2)
            x_hat = np.squeeze(x_hat)
            x_hat_2 = np.squeeze(x_hat_2)
        absolute_diff = abs(f1 - f2)
    elif version == '2':
        f1, f2 = 0, 0
        for i in range(len(X)):
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
    else:
        for j in indices_baseline:
            x_hat[j] = baseline[j]
            x_hat_2[j] = baseline[j]
        x_hat = np.reshape(x_hat, (1, N))
        f1 = model.predict_proba(x_hat)[0][1] if is_classification else model.predict(x_hat)
        x_hat_2[xi] = baseline[xi]
        x_hat_2 = np.reshape(x_hat_2, (1, N))
        f2 = model.predict_proba(x_hat_2)[0][1] if is_classification else model.predict(x_hat_2)
        absolute_diff = abs(f1 - f2) / len_X
    # print("V1- Abs: ", f1, f2)
    return absolute_diff, f1, f2


def approximate_shapley(xi, N, X, x, m, model, baseline, unique_count, is_classification, global_shap=False):
    R = list(itertools.permutations(range(N)))
    random.shuffle(R)
    score = 0

    count_negative = 0
    for i in range(m):
        r = list(R[i])
        xi_index = r.index(xi)
        s_features = r[:xi_index + 1]
        s_hat_features = r[xi_index + 1:]

        abs_diff, f1, f2 = get_expectation(X, x, s_features, s_hat_features, baseline, unique_count, model, N,
                                           is_classification, xi)
        score = score + abs_diff
    if not global_shap:
        if f2 > f1:
            count_negative -= 1
        else:
            count_negative += 1
        if count_negative < 0:
            score = -1 * score
    return score / m


def main(file_name='synthetic1', local_shap=0, global_shap=True, is_classification=False):
    sigma_phi = 0
    df = pd.read_csv('../output/dataset/' + file_name + '.csv')
    print(df.columns)
    model = pickle.load(open('../output/model/' + file_name + '.sav', 'rb'))

    n_features = len(df.columns[:-1])
    X = df.iloc[:, :n_features].values
    unique_count = collections.Counter(map(tuple, X))
    ##### f(x) with baseline
    f_o = get_baseline(X, model)
    baseline = np.mean(X, axis=0)
    # To convert baseline (mean) to discrete
    if is_classification:
        baseline = [1 if i > 0.5 else 0 for i in baseline]
    print("Baseline f(x): ", f_o)

    # x = X[:10]
    if global_shap:
        global_shap_score = 0
        for feature in range(n_features):
            for x in X:
                global_shap_score += approximate_shapley(feature, n_features, X, x, mt.factorial(n_features), model,
                                                         baseline, unique_count, is_classification, global_shap)
            global_shap_score = global_shap_score / len(X)
            print("x", str(feature + 1), ": ", global_shap_score)
    # print(global_shap_score)
    else:
        x = X[local_shap]
        local_shap_score = 0
        for feature in range(n_features):
            local_shap_score = approximate_shapley(feature, n_features, X, x, mt.factorial(n_features), model, baseline,
                                                   unique_count,
                                                   is_classification)
            # local_shap_score = local_shap_score / len(X)
            print("x", str(feature + 1), ": ", local_shap_score)
            sigma_phi += local_shap_score
        x = np.reshape(x, (1, n_features))
        print("local f(x): ", model.predict(x))
        print("Sigma_phi: ", sigma_phi)


def test(file_name='synthetic1'):
    df = pd.read_csv('../output/dataset/' + file_name + '.csv')
    print(df.columns)
    model = pickle.load(open('../output/model/' + file_name + '.sav', 'rb'))

    n_features = len(df.columns[:-1])
    X = df.iloc[:, :n_features].values
    fx = 0
    for i in X:
        x = np.reshape(i, (1, n_features))
        fx += model.predict(x)
    print("E(fx): ", fx / len(X))
    ##### f(x) with baseline
    baseline = np.mean(X, axis=0)
    baseline = np.reshape(baseline, (1, n_features))
    print("f(Ex): ", model.predict(baseline))


main(file_name='synthetic_discrete', local_shap=16, is_classification=True, global_shap=False)
# test(file_name='synthetic2')
