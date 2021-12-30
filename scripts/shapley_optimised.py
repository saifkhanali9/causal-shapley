import os
import pickle

import time
import numpy as np
import pandas as pd
import math
import seaborn as sns
from joblib import load, Parallel, delayed
from matplotlib import pyplot as plt
import json
import torch.nn as nn
import torch
from autoencoder import TorchDecoder, TorchEncoder

if torch.cuda.is_available():
    #     dev = "cuda:0"
    dev = "cpu"
else:
    dev = "cpu"
print(dev)


def calc_permutations(S, p):
    return (math.factorial(len(S)) * math.factorial(p - len(S) - 1)) / math.factorial(p)


def show_values(axs, orient="h", space=0):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def predict(x, model):
    return pred(x, model[0], model[1])


def pred(x, enc, dec):
    if len(x.shape) == 1:
        x = torch.reshape(x, (1, -1))
    enc.eval()
    dec.eval()
    # x = x[1]
    y = enc(x)
    y = dec(y)
    loss_fn = nn.MSELoss(reduction='none')
    los_val = loss_fn(y, x)
    loss_per_row = torch.mean(los_val, dim=1)
    loss_per_row[loss_per_row > loss_threshold] = 1
    loss_per_row[loss_per_row <= loss_threshold] = 0
    ll = loss_per_row.unique(return_counts=True)
    return loss_per_row


def get_baseline(X, model):
    fx = torch.mean(predict(X, model))
    return fx.item()


def powerset(features):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    listrep = list(features)
    n = len(listrep)
    return [[listrep[k] for k in range(n) if i & 1 << k] for i in range(2 ** n)]


def baseline(X, x, features_baseline, model):
    temp_row = torch.zeros(X.shape)
    temp_row[:] = x
    temp_row[:, features_baseline] = X[:, features_baseline]
    f1 = torch.mean(predict(temp_row, model))
    return f1


def multithreaded_powerset(X, x, s, s_index, feature_index, features_list, model):
    # for count_power, s in enumerate(S):
    print(s_index, '/', 2 ** len(x))
    s_baseline = list(set(s).symmetric_difference(features_list))
    s_union_j_baseline = s_baseline[:]
    s_union_j_baseline.remove(feature_index)
    v_u_j = baseline(X, x, s_union_j_baseline, model)
    v = baseline(X, x, s_baseline, model)
    # phi_i += calc_permutations(S=s, p=X.shape[1]) * (v_u_j - v)
    return calc_permutations(S=s, p=X.shape[1]) * (v_u_j - v)


def multithreaded_main(feature, X, x, features_list, model):
    # for i in features_list:
    start_time = time.time()
    features_list_copy = features_list[:]
    del features_list_copy[feature]
    S = powerset(features_list_copy)
    phi_i = 0
    for count_power, s in enumerate(S):
        print(feature, count_power, '/', 2 ** (len(features_list) - 1))
        s_baseline = list(set(s).symmetric_difference(features_list))
        s_union_j_baseline = s_baseline[:]
        s_union_j_baseline.remove(feature)
        v_u_j = baseline(X, x, s_union_j_baseline, model)
        v = baseline(X, x, s_baseline, model)
        phi_i += calc_permutations(S=s, p=X.shape[1]) * (v_u_j - v)
    phi_i = phi_i.item()
    diff_time = round(time.time() - start_time, 3)
    # total_time += diff_time
    print("Feature", feature, " | ", feature_names[feature], ": \t\t", round(phi_i, 4), '\t Time taken: ', diff_time,
          'sec')
    return phi_i


def shap_optimized(X, x, feature_names, model):
    features_list = list(range(len(feature_names)))
    feature_scores = []
    # Loop for phi_i
    total_time = 0
    feature_scores = Parallel(n_jobs=-1)(
        delayed(multithreaded_main)(feature, X, x, features_list, model) for feature in
        features_list)
    print(feature_scores, type(feature_scores))
    print('\nTotal time: ', round(total_time, 4), ' sec\n\nBaseline: ', baseline_V, '\nSigma_phi: ',
          sum(feature_scores))
    # Making use of Sigma_phi = f(x) + f_o
    # Where f_o = E(f(X))
    print("\nlocal f(x):\t\t\t", predict(x, model), "\nSigma_phi + E(fX):\t",
          round(sum(feature_scores) + baseline_V, 7))
    return feature_scores


# Setting parameters/paths and loading files
file_name = 'census/'
anomaly_type = 'distance_sex_hpw_workclass'
file_path = '../output/anomaly_included/' + file_name + anomaly_type
x_path = file_path + '/x_test.csv'
anomaly_path = file_path + '/anomalous_data.csv'
output_file_path = file_path + '/explanations/'
anomalous = '../output/anomaly_included/'
is_classification = True
local_index = 15
epoch = 1000
loss_threshold = 210

# Loading/Reading files
model_path = '/home/saif/MyStuff/UdS/Thesis/Proposal/causal-shapley/output/model/census2/all_epochs_no_dropouts/'
encoder_model = model_path + 'ep_' + str(epoch) + '_encoder_model.pth'
decoder_model = model_path + 'ep_' + str(epoch) + '_decoder_model.pth'
df = pd.read_csv(x_path)
anomalous_data = pd.read_csv(anomaly_path).to_numpy()
feature_names = df.columns.tolist()
X = df.to_numpy()
print(X.shape)
os.makedirs(output_file_path, exist_ok=True)
encoder = TorchEncoder(in_dim=X.shape[1]).to(dev)
decoder = TorchDecoder(out_dim=X.shape[1]).to(dev)
encoder.load_state_dict(torch.load(encoder_model))
decoder.load_state_dict(torch.load(decoder_model))
encoder.eval()
decoder.eval()
model = [encoder, decoder]
X = torch.tensor(X, dtype=torch.float).to(dev)
# Operations Starting
baseline_V = get_baseline(X, model)

for row_num, data_point in enumerate(anomalous_data):
    data_point = torch.tensor(data_point, dtype=torch.float).to(dev)
    explanations = {}
    feature_score_list = shap_optimized(X, data_point, feature_names, model)

    # Plotting and saving explanations
    for i, name in enumerate(feature_names):
        explanations[name] = feature_score_list[i]
    explanations = dict(sorted(explanations.items(), key=lambda item: item[1]))
    with open(output_file_path + str(row_num) + '.txt', 'w') as file:
        file.write(json.dumps(explanations))
    plot = sns.barplot(y=feature_names, x=feature_score_list)
    plt.savefig(output_file_path + str(row_num) + '.png')
    break
