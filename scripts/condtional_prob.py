import pandas as pd
import numpy as np


def get_probs(k, index, invert=False):
    length = len(k[0])
    if len(k) == 1:
        if invert:
            count = np.count_nonzero(k[0] == 0)
        else:
            count = np.count_nonzero(k[0] == 1)
        return count / length
    else:
        count = 0
        for i in range(len(k[0])):
            check_condition = 1
            for j in range(len(k)):
                check_condition *= k[j][i]
                if j == index and invert:
                    if k[j][i] == 0:
                        check_condition = 1
                    else:
                        check_condition = 0
                if check_condition == 0:
                    break
            count += check_condition
        return count / length


def concat_list(joint):
    joint_2 = []
    for i in joint:
        for j in i:
            joint_2.append(j)
    return joint_2


def conditional(a, b, c):
    joint = concat_list([a, b, c])
    den = concat_list([b, c])
    prob = 0
    invert_index = [False, True]
    for i in invert_index:
        numerator = get_probs(joint, index=2, invert=i)
        denominator = get_probs(den, index=1, invert=i)
        kk = get_probs(c, index=0, invert=i) if True in invert_index else 1
        prob += (numerator / denominator) * kk
    return prob


def causal():
    df = pd.read_csv('../output/dataset/synthetic_discrete' + '.csv')
    X = df.iloc[:, :].values
    A = X[:, 0]
    B = X[:, 1]
    Y = X[:, 4]
    return conditional([Y], [B], [A])


def temp():
    df = pd.read_csv('../output/dataset/synthetic_discrete' + '.csv')
    X = df.iloc[:, :].values
    A = X[:, 0]
    B = X[:, 1]
    Y = X[:, 4]
    count = 0
    for i in range(len(A)):
        if B[i] == 1 and A[i] == 0:
            count += 1
    print(count / len(A))


# conditional('synthetic_discrete')
print(causal())
# temp()
