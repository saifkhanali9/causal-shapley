import numpy as np
import pandas as pd
import itertools

from joblib import Parallel, delayed
from scipy.spatial import distance
from numpy.linalg import norm

unique_values = {}


def parallelized_distance(index, i, X):
    print(index)
    dist = min(np.linalg.norm(i - X, axis=1))
    return np.append(i, dist)


def prob(X_new, permutation, features, parent_child=True):
    permutation = np.array(permutation.tolist())
    temp = np.zeros((permutation.shape[0], 3))
    if parent_child:
        for index in range(len(permutation[0]) - 1):
            feature_used = [features[index], features[-1]]
            print("index", index)
            permutation_new = permutation[:, [index, -1]]

            X = X_new[feature_used].values
            temp_list = []
            for index, i in enumerate(permutation_new):
                dist = min(np.linalg.norm(i - X, axis=1))
                temp[index] = np.append(i, dist)
            temp = temp[temp[:, -1].argsort()]
            for i in temp[-10:]:
                print('\tCount:', i)
    else:
        feature_used = features
        # X = X_new[feature_used].values.tolist()
        X = X_new[feature_used].values
        # X.astype(float)
        temp = np.zeros((permutation.shape[0], permutation.shape[1] + 1))
        temp_list = []
        # for index, i in enumerate(permutation):
        #     print(index)
        #     dist = min(np.linalg.norm(i - X, axis=1))
        #     temp[index] = np.append(i, dist)
        temp = Parallel(n_jobs=-1)(delayed(parallelized_distance)(index, i, X) for index, i in enumerate(permutation))
        temp = np.array(temp)
        temp = temp[temp[:, -1].argsort()]
        temp = temp[-100:]
        for i in temp:
            print('\tCount:', i)


def calc_prob(perm, features):
    all_list = []
    for j in features:
        all_list.append(perm[j])
    res = list(itertools.product(*all_list))
    return res


def permutation(X, key):
    if key not in unique_values:
        unique_values[key] = sorted(X[key].unique())


# causal = {
#     "race": ["education", "marital_status", "occupation"],
#     "native_country": ["education", "relationship"],
#     "sex": ["education", "relationship", "occupation"]
# }
causal = {
    "age": ["hours_per_week", "sex"]
}
x_train = pd.read_csv('../output/dataset/census/x_train.csv')
x_test = pd.read_csv('../output/dataset/census/x_test.csv')
for i in causal:
    causal[i].append(i)
    features = causal[i]
for j in features:
    permutation(x_train, j)
permutations = calc_prob(unique_values, features)
# print(i)
prob(x_train, np.array(permutations), features, parent_child=False)
