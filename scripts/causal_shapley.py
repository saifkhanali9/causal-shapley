import itertools
import pickle
import numpy as np
import pandas as pd
import random
import copy
import math as mt
from condtional_prob import conditional

random.seed(42)


def get_baseline(X, model):
    fx = 0
    n_features = X.shape[1]
    X = np.reshape(X, (len(X), 1, n_features))
    for i in X:
        # x = np.reshape(i, (1, n_features))
        fx += model.predict(i)
    return fx / len(X)


def get_expectation(X, x, baseline, indices, indices_baseline, model, N, is_classification, xi):
    x_hat = np.zeros(N)
    for j in indices:
        x_hat[j] = x[j]
    # for j in indices_baseline:
    #     x_hat[j] = baseline[j]
    f1, f2 = 0, 0
    for i in range(len(X)):
        for j in indices_baseline:
            x_hat[j] = 
            x_hat = np.reshape(x_hat, (1, N))
            f1 = model.predict_proba(x_hat)[0][1] if is_classification else model.predict(x_hat)
            x_hat[0][xi] = baseline[xi]
            f2 = model.predict_proba(x_hat)[0][1] if is_classification else model.predict(x_hat)
    return abs(f1 - f2), f1, f2


def approximate_shapley(xi, N, X, x, m, model, baseline, is_classification, global_shap=False):
    R = list(itertools.permutations(range(N)))
    random.shuffle(R)
    score = 0

    count_negative = 0
    for i in range(m):
        # baseline = random.choice(X)
        r = list(R[i])
        # print(r.index(xi))
        xi_index = r.index(xi)
        s_features = r[:xi_index + 1]
        s_hat_features = r[xi_index + 1:]

        # print("\n\nr: ", r, "S: ", s_features, "S_hat: ", s_hat_features)
        # for j in s_features:
        #     x_hat[j] = x[j]
        # for j in s_hat_features:
        #     x_hat[j] = baseline[j]
        # x_hat = np.reshape(x_hat, (1, N))
        #
        # f1 = get_expectation(X, x_hat, baseline, s_features, s_hat_features, model, N, is_classification, xi)
        # # print("x: ",  x, "\t", "base_line: ", baseline, "\tx_hat: ", x_hat)
        #
        # x_hat_2 = x_hat
        # x_hat_2[0][xi] = baseline[xi]
        # f2 = model.predict_proba(x_hat_2)[0][1] if is_classification else model.predict(x_hat_2)
        abs_diff, f1, f2 = get_expectation(X, x, baseline, s_features, s_hat_features, model, N,
                                           is_classification, xi)
        score = score + abs_diff
        # print("x_hat_2: ", x_hat_2)
    if not global_shap:
        if f2 > f1:
            count_negative -= 1
        else:
            count_negative += 1
        if count_negative < 0:
            score = -1 * score
    return score / m


def main(file_name='synthetic1', local_shap=0, global_shap=True, is_classification=False):
    df = pd.read_csv('../output/dataset/' + file_name + '.csv')
    print(df.columns)
    model = pickle.load(open('../output/model/' + file_name + '.sav', 'rb'))

    n_features = len(df.columns[:-1])
    X = df.iloc[:, :n_features].values

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
                                                         baseline, is_classification, global_shap)
            global_shap_score = global_shap_score / len(X)
            print("x", str(feature + 1), ": ", global_shap_score)
    # print(global_shap_score)
    else:
        x = X[local_shap]
        local_shap_score = 0
        for feature in range(n_features):
            local_shap_score = approximate_shapley(feature, n_features, X, x, mt.factorial(n_features), model, baseline,
                                                   is_classification)
            # local_shap_score = local_shap_score / len(X)
            print("x", str(feature + 1), ": ", local_shap_score)
        x = np.reshape(x, (1, n_features))
        print("local f(x): ", model.predict(x))


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


main(file_name='synthetic2', local_shap=12, is_classification=False, global_shap=False)
# test(file_name='synthetic2')
