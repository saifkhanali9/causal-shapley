import math as mt
import os
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd
from numpy.random import rand
import random

random.seed(42)


def _indices(shapes):
    idx = [0] * len(shapes)
    yield idx[::]
    while True:
        for i in range(len(shapes) - 1, -1, -1):
            idx[i] += 1
            if idx[i] == shapes[i]:
                if i == 0:
                    return
                idx[i] = 0
                continue
            else:
                yield idx[::]
                break


def _independent_dataset(sub_categories):
    D = []
    for i in _indices(sub_categories):
        D.append(i)
    return np.array(D, dtype=np.uint8)


def _sample_atleastonce(P, n_instances):
    prob_vector = P.flatten()
    indices = list(range(len(prob_vector)))
    q_vector = []

    # To handle zero Probabilities. (We'll get second minimum prob if 0 exists)
    unique_p = list(set(prob_vector))
    unique_p.sort()
    min_p = unique_p[0]
    if unique_p[0] == 0:
        min_p = unique_p[1]
    for i in prob_vector:
        q_vector.append(round(i / min_p))
    sampled = []

    # If q vector alone isn't sufficient
    if sum(q_vector) < n_instances:
        sampled = np.random.choice(indices, n_instances - sum(q_vector), p=prob_vector)

    # If you want to sample less than what the p vector suggests
    elif sum(q_vector) > n_instances:
        q_vector = []
        sampled = np.random.choice(indices, n_instances, p=prob_vector)
    return q_vector, sampled


def _sample(n_instances, P):
    sub_categories = list(P.shape)
    # Creates all possible combinations
    D = _independent_dataset(sub_categories)

    # Samples out according to prob tensor
    q_vector, sampled_indices = _sample_atleastonce(P, n_instances)
    D_sampled = []
    if len(q_vector) > 0:
        for i in range(len(q_vector)):
            for j in range(q_vector[i]):
                D_sampled.append(D[i])
    if len(sampled_indices) > 0:
        for i in sampled_indices:
            D_sampled.append(D[i])
    D_sampled = np.array(D_sampled)
    return D_sampled


def _create_pmatrix(instances, sub_categories):
    total_possibilities = np.prod(sub_categories)
    len_p = np.prod(sub_categories)
    rnd_array = np.random.multinomial(instances, np.ones(len_p) / len_p, size=1)[0]
    rnd_array.sort()
    while True:
        if 0 in rnd_array and instances >= total_possibilities:
            rnd_array.sort()
            if rnd_array[-1] > 1:
                rnd_array[0] = 1
                rnd_array[-1] -= 1
                continue
        else:
            random.shuffle(rnd_array)
            p_vector = rnd_array / sum(rnd_array)
            p_adjusted = 1 - sum(p_vector)
            p_vector[0] += p_adjusted
            sub_categories = tuple(sub_categories)
            P = np.reshape(p_vector, sub_categories)
            return P


def _induce_noise(x):
    noise_index = list(dict.fromkeys([random.randrange(1, len(x)) for _ in range(int(len(x) * 0.05))]))
    new_x = [0] * len(x)
    for i in range(len(x)):
        new_x[i] = x[i]
        if i in noise_index:
            new_x[i] = 1 if x[i] == 0 else 0
    return new_x


def _add_features(D, file_name):
    D = np.array(D)
    x1 = D[:, 0]
    x2 = _induce_noise(x1)
    x3 = D[:, 1]
    x4 = _induce_noise(x3)
    y = []
    for i in range(len(x1)):
        y.append(x2[i] * x1[i])
    # _induce_noise(xi)
    # x2[noise_index] = 0
    # print(noise_index)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})
    df.to_csv(Path('../output/dataset/') / file_name, index=False)
    # return


def gen_desc(n_instances, sub_categories, file_name='synthetic_discrete_3.csv'):
    """
    Generates dataset directly without having to save a json file.
    Args:
        n_instances (int): Total number of instances (rows) to be created:
        sub_categories (list/array): List representing subcategories for each feature. E.g in [2, 3, 2] first and second
        features are binary, while second feature is of 3 categories and resulting P will be a 3d tensor in this case.

    Returns:
        Np.Array: Returns generated dataset from parameters.
    """
    P = _create_pmatrix(n_instances, sub_categories)
    D = _sample(n_instances, P)
    # _save_dataset(D, 'discrete')
    _add_features(D, file_name)
    # return D


def gen_dataset(n_instances=1000, file_name='synthetic'):
    try:
        os.mkdir('../dataset/')
    except FileExistsError:
        pass
    x1 = np.random.randn(n_instances) * 10
    x2 = x1 + np.random.randn(n_instances)
    x4 = np.random.randn(n_instances) + 1000
    x3 = x4 + np.random.randn(n_instances)
    y = x1 + np.random.randn(n_instances) + x2
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    y = y.to_numpy()
    y = np.reshape(y, (len(y), 1))
    discretizer.fit(X)
    discrete_X = discretizer.transform(X)
    discrete_dataset = np.append(discrete_X, y, axis=1)
    df.to_csv('../output/dataset/' + file_name + '.csv', index=False)
    discrete_df = pd.DataFrame(data=discrete_dataset, columns=['x1', 'x2', 'x3', 'x4', 'y'])
    discrete_df.to_csv('../output/dataset/' + file_name + '_disc.csv', index=False)
    test = 0


gen_dataset(n_instances=30000, file_name='synthetic_cont_2')

# print(gen_desc(30000, [2, 2]))
