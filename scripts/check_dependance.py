import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

from autoencoder import TorchEncoder, TorchDecoder

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
print(dev)


def pred(x, enc, dec):
    # x = x[1]
    y = enc(x)
    y = dec(y)
    loss_fn = nn.MSELoss(reduction='none')
    los_val = loss_fn(y, x)
    return torch.mean(los_val, dim=1).cpu().detach().numpy()


def dependence():
    data_path = '../datasets/prepared'
    x = pd.read_csv(data_path + '/x_train.csv')
    try:
        os.makedirs(data_path + '/misc')
    except FileExistsError:
        pass
    continuous_features = ['age', 'education', 'education_num', 'hours_per_week']
    x = x[continuous_features]
    methods = ['pearson', 'kendall', 'spearman']
    for method in methods:
        df = x.corr(method=method)
        print(method, ':\n', df)
        df.to_csv(data_path + '/misc/dependence_' + method + '.csv')


# To generate Anomaly
def corr_cont(x, model, features=None, keep_max=False):
    if features is None:
        features = ["age", "education"]
    replaced_feature_index_keep_max = x.columns.tolist().index(features[1])
    replaced_feature_index_keep_min = x.columns.tolist().index(features[0])
    x_sorted = x.sort_values(features[0])
    anoamlous_data = x_sorted.tail(total_anomalies).to_numpy()
    row_min = x_sorted.iloc[x_sorted[features[1]].idxmin()]
    anoamlous_data[:, replaced_feature_index_keep_min] = int(row_min[features[1]])
    randomlist = random.sample(range(0, x.shape[0]), total_anomalies)
    normal_loss = pred(torch.tensor(x.to_numpy(), dtype=torch.float).to(dev), model[0], model[1])
    anoamlous_loss = pred(torch.tensor(anoamlous_data, dtype=torch.float).to(dev), model[0], model[1])
    plt.scatter(list(range(len(normal_loss))), normal_loss)
    plt.scatter(randomlist, anoamlous_loss)
    plt.show()
    ll = 'ee'
    # for row in anoamlous_data:
    #
    # row_max = x.iloc[x[features[0]].idxmax()]
    # row_min = x.iloc[x[features[1]].idxmin()]
    # if keep_max:
    #     row_max = row_max.to_list()
    #     row_max[replaced_feature_index_keep_max] = int(row_min[features[1]])
    # else:
    #     row_min = row_min.to_list()
    #     row_min[replaced_feature_index_keep_min] = int(row_max[features[0]])


epoch = 1000
total_anomalies = 130
parent_path = '../output/anomaly_included'
dataset_name = 'census'
anomaly_type = 'distance_sex_hpw_workclass'
file_path = f'{parent_path}/{dataset_name}/{anomaly_type}'
X = pd.read_csv(f'{file_path}/x_train.csv')
model_path = '../output/model/census2/all_epochs_no_dropouts/'
x_normal = pd.read_csv('../output/dataset/census/x_test.csv').to_numpy()
encoder_model = model_path + 'ep_' + str(epoch) + '_encoder_model.pth'
decoder_model = model_path + 'ep_' + str(epoch) + '_decoder_model.pth'
encoder = TorchEncoder(in_dim=X.shape[1]).to(dev)
decoder = TorchDecoder(out_dim=X.shape[1]).to(dev)
encoder.load_state_dict(torch.load(encoder_model))
decoder.load_state_dict(torch.load(decoder_model))
model = [encoder, decoder]
corr_cont(X, model)
