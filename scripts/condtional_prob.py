import numpy as np

# Probability is taken over indices of baseline only
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


def causal_prob(unique_count, x_hat, indices, indices_baseline, causal_struc, n):
    p = 1
    for i in indices_baseline:
        intersect_s, intersect_s_hat = [], []
        intersect_s_hat.append(i)
        if causal_struc[i] != None:
            if causal_struc[i] in indices or causal_struc[i] in indices_baseline:
                intersect_s.append(causal_struc[i])
            p *= conditional_prob(unique_count, x_hat, intersect_s, intersect_s_hat, n)
        else:
            p *= conditional_prob(unique_count, x_hat, [], intersect_s_hat, n)
    return p