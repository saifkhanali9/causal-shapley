import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

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
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    los_val = loss_fn(y, x)
    return torch.mean(los_val, dim=1).cpu().detach().numpy()


epoch = 227
model_path = '../output/model/census/all_epochs/'
x_normal = pd.read_csv('../output/dataset/census/x_train.csv').to_numpy()
encoder_model = model_path + 'ep_' + str(epoch) + '_encoder_model.pth'
decoder_model = model_path + 'ep_' + str(epoch) + '_decoder_model.pth'
encoder = TorchEncoder(in_dim=x_normal.shape[1]).to(dev)
decoder = TorchDecoder(out_dim=x_normal.shape[1]).to(dev)
encoder.load_state_dict(torch.load(encoder_model))
decoder.load_state_dict(torch.load(decoder_model))


print(x_normal.shape)
normal_loss = pred(torch.tensor(x_normal, dtype=torch.float).to(dev), encoder, decoder)
print(normal_loss, max(normal_loss))

x_anomaly = np.loadtxt('../notebooks/age_hpw_anomaly.txt')
anomalous_loss = pred(torch.tensor(x_anomaly, dtype=torch.float).to(dev), encoder, decoder)
print(anomalous_loss)
plt.scatter(list(range(len(normal_loss))), normal_loss)
# plt.scatter(list(range(len(normal_loss))), anomalous_loss)
# plt.savefig(plot_path + '/all_anomalies' + '.png')
plt.show()