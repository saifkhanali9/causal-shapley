import itertools
import pickle
import numpy as np
import pandas as pd
import random
import copy
import math as mt
random.seed(42)

def get_baseline(X, model):
    fx = 0
    n_features = X.shape[1]
    X = np.reshape(X, (len(X), 1, n_features))
    for i in X:
        # x = np.reshape(i, (1, n_features))
        fx += model.predict(i)
    return fx / len(X)


def approximate_shapley(xi, N, x, m, model, baseline, is_classification, global_shap=False):
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
        x_hat = np.zeros(N)

        # print("\n\nr: ", r, "S: ", s_features, "S_hat: ", s_hat_features)
        for j in s_features:
            x_hat[j] = x[j]
        for j in s_hat_features:
            x_hat[j] = baseline[j]
        x_hat = np.reshape(x_hat, (1, N))

        f1 = model.predict_proba(x_hat)[0][1] if is_classification else model.predict(x_hat)
        # print("x: ",  x, "\t", "base_line: ", baseline, "\tx_hat: ", x_hat)

        x_hat_2 = x_hat
        x_hat_2[0][xi] = baseline[xi]
        f2 = model.predict_proba(x_hat_2)[0][1] if is_classification else model.predict(x_hat_2)
        score = score + abs(f1 - f2)
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
    # if is_classification:
    #     baseline = [1 if i > 0.5 else 0 for i in baseline]
    print("Baseline f(x): ", f_o)

    # x = X[:10]
    if global_shap:
        global_shap_score = 0
        for feature in range(n_features):
            for x in X:
                global_shap_score += approximate_shapley(feature, n_features, x, mt.factorial(n_features), model,
                                                         baseline, is_classification, global_shap)
            global_shap_score = global_shap_score / len(X)
            print("x", str(feature + 1), ": ", global_shap_score)
    # print(global_shap_score)
    else:
        sigma_phi = 0
        x = X[local_shap]
        local_shap_score = 0
        for feature in range(n_features):
            local_shap_score = approximate_shapley(feature, n_features, x, mt.factorial(n_features), model, baseline,
                                                   is_classification)
            # local_shap_score = local_shap_score / len(X)
            print("x", str(feature + 1), ": ", local_shap_score)
            sigma_phi += local_shap_score
        x = np.reshape(x, (1, n_features))
        print("local f(x): ", model.predict(x))
        print("Sigma_phi + E(fX): ", round(sigma_phi + f_o[0], 3))



main(file_name='synthetic_discrete_2', local_shap=15, is_classification=True, global_shap=False)
# main(file_name='synthetic2', local_shap=12, is_classification=False, global_shap=False)
# test(file_name='synthetic_discrete')
