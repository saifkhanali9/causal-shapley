import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import seaborn as sns

from autoencoder import TorchEncoder, TorchDecoder

if torch.cuda.is_available():
    # dev = "cuda:0"
    dev = "cpu"
else:
    dev = "cpu"
print(dev)


def pred(x, enc, dec):
    y = enc(x)
    y = dec(y)
    loss_fn = nn.MSELoss(reduction='none')
    los_val = loss_fn(y, x)
    return torch.mean(los_val, dim=1).cpu().detach().numpy()


# Setting parameters/paths and loading files
file_name = 'census2/'
file_path = '../output/dataset/' + file_name
x_normal = pd.read_csv('../output/anomaly_included/census/tuner_hpw_age/' + 'x_test.csv').to_numpy()

distance_anomaly = pd.read_csv(
    '../output/anomaly_included/census/' + 'distance_sex_hpw_workclass/anomalous_data.csv').to_numpy()
tuner_anomaly = pd.read_csv('../output/anomaly_included/census/' + 'jptemp/anomalous_data.csv').to_numpy()
epoch = 1000
total_anomalies = 100

# Loading/Reading files
model_path = '../output/model/census2/all_epochs_no_dropouts/'
encoder_model = model_path + 'ep_' + str(epoch) + '_encoder_model.pth'
decoder_model = model_path + 'ep_' + str(epoch) + '_decoder_model.pth'
encoder = TorchEncoder(in_dim=x_normal.shape[1]).to(dev)
decoder = TorchDecoder(out_dim=x_normal.shape[1]).to(dev)
encoder.load_state_dict(torch.load(encoder_model))
score_normal = encoder(torch.tensor(x_normal, dtype=torch.float).to(dev)).detach().numpy()
score_tuner = encoder(torch.tensor(tuner_anomaly, dtype=torch.float).to(dev)).detach().numpy()
f1 = score_normal[:, 0]
f2 = score_normal[:, 1]
f3 = score_normal[:, 2]
a1 = score_tuner[:, 0]
a2 = score_tuner[:, 1]
a3 = score_tuner[:, 2]
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(f1, f2, f3)
ax.scatter(a1, a2, a3)
plt.show()


tt = 'f'
# score_normal = pred(torch.tensor(x_normal, dtype=torch.float).to(dev), encoder, decoder)
#
# score_distance = pred(torch.tensor(distance_anomaly, dtype=torch.float).to(dev), encoder, decoder)
# score_tuner = pred(torch.tensor(tuner_anomaly, dtype=torch.float).to(dev), encoder, decoder)
# sns.scatterplot(y=score_normal, x=list(range(len(score_normal))))
# randomlist = random.sample(range(0, len(score_normal)), total_anomalies)
# sns.scatterplot(y=score_distance, x=randomlist)
# randomlist = random.sample(range(0, len(score_normal)), total_anomalies)
# sns.scatterplot(y=score_tuner, x=randomlist)
# plt.show()
