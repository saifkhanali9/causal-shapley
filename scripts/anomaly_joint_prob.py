import numpy as np
import pandas as pd
import itertools

unique_values = {}


def prob(X_new, permutation, features, parent_child=True):
    # permutation = permutation.tolist()
    if parent_child:
        for index in range(len(features)):
            feature_used = [features[index], features[-1]]
            print("index", index)
            permutation_selected = permutation[:, [index, -1]]
            # Removing duplication that is caused because we're taking subset
            permutation_list = np.unique(permutation_selected, axis=0).tolist()
            print("For features ", features[index], features[-1])
            X = X_new[feature_used]
            # Making copy of permutation
            X = X.values.tolist()
            temp = np.zeros((len(permutation_list), len(permutation_list[0]) + 1))
            if index == len(features) - 1:
                break
            for index2, i in enumerate(permutation_list):
                temp[index2, :2] = np.array(i)
                temp[index2, -1] = X.count(i)
                # print(i, '\tCount:', count)
            temp = temp[temp[:, -1].argsort()]
            for i in temp[:10]:
                print('\tCount:', i)
    else:
        feature_used = features
        X = X_new[feature_used].values.tolist()
        for i in permutation:
            count = X.count(i)
            print(i, '\tCount:', count)


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
    "hours_per_week": ["education_num", "education"]
}
x_train = pd.read_csv('../output/dataset/census2/x_train.csv')
# x_test = pd.read_csv('../datasets/prepared/x_test.csv')
for i in causal:
    causal[i].append(i)
    for j in causal[i]:
        permutation(x_train, j)
    permutations = calc_prob(unique_values, causal[i])
    prob(x_train, np.array(permutations), causal[i], parent_child=True)
