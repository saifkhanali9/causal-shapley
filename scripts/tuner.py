from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
import os
import shutil
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.auto_encoder import AutoEncoder
# import sys
# sys.path.append('../scripts')
from autoencoder import TorchEncoder, TorchDecoder
import random
import torch
import torch.nn as nn
import seaborn as sns

if torch.cuda.is_available():
    #     dev = "cuda:0"
    dev = "cpu"
else:
    dev = "cpu"

x_train = pd.read_csv('../output/dataset/census2/x_test.csv')
x1_name = 'age'
x2_name = 'hours_per_week'
tuner = 2.2

X = x_train[x1_name]
y = x_train[x2_name]
X_seq = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
coefs = np.polyfit(X.values.flatten(), y.values.flatten(), 5)
coefs_new = np.copy(coefs)
coefs_new *= tuner


def std(X_, x1, x_name=x1_name, y_name=x2_name):
    X__ = X_[X_[x_name] == x1]
    mu = np.mean(X__)
    sigma = np.std(X__)
    #     print(mu)
    return mu[y_name], sigma[y_name]


y_new = []
for i in X:
    mu, sigma = std(x_train, i)
    noise = np.random.normal(mu, sigma, 1)[0]
    y_new.append(np.polyval(coefs_new, i) + noise)


def pred(x, enc, dec):
    # x = x[1]
    y = enc(x)
    y = dec(y)
    loss_fn = nn.MSELoss(reduction='none')
    # print("Kinaan k tattey: ",y)
    los_val = loss_fn(y, x)
    # print(los_val)
    return torch.mean(los_val, dim=1).cpu().detach().numpy()


total_anomalies = 200
epoch = 1000
df_normal = pd.read_csv('../output/dataset//census2/x_test.csv')
model_path = '../output/model/census2/all_epochs_no_dropouts/'
encoder_model = model_path + 'ep_' + str(epoch) + '_encoder_model.pth'
decoder_model = model_path + 'ep_' + str(epoch) + '_decoder_model.pth'
anomaly_desciption = f'Feature hpw is regressed over feature age with tuner value ' + str(tuner) + '.'

y_new = np.array(y_new)
x_anomalous = df_normal.copy()
# print(type(x_anomalous))
x_anomalous[x2_name] = y_new
x_anomalous = x_anomalous.sample(n=total_anomalies)
# print(x_train.head())
# x_new_2 = x_train_anomalous.to_numpy()
# if True:
dataset_path = '../output/anomaly_included_1000/census/'
anomaly_type = 'tuner'
anomaly_name = '_hpw_age/'
anomaly_path = dataset_path + anomaly_type + anomaly_name
os.makedirs(anomaly_path, exist_ok=True)
with open(anomaly_path + 'readme.txt', 'w') as f:
    f.write(anomaly_desciption)

# Train set
shutil.copyfile('../output/dataset/census2/x_train.csv', anomaly_path + '/x_train.csv')

# Antoine's set
df_normal[:1000].to_csv(anomaly_path + 'x_test_antoine.csv', index=False)
# Test set
df_normal[1000:].to_csv(anomaly_path + 'x_test.csv', index=False)
# Anomalies
# x_anomalous[:-30].to_csv(anomaly_path + 'anomalous_data.csv', index=False)
# # Anomalies Antoine
# x_anomalous[-30:].to_csv(anomaly_path + 'anomalous_data_antoine.csv', index=False)


# print(x_normal.shape)


# x_anomalous = x_anomalous.to_numpy()
x_normal = df_normal.to_numpy()
encoder = TorchEncoder(in_dim=x_normal.shape[1]).to(dev)
decoder = TorchDecoder(out_dim=x_normal.shape[1]).to(dev)
encoder.load_state_dict(torch.load(encoder_model))
decoder.load_state_dict(torch.load(decoder_model))

# x_anomalous = df_anomalous.to_numpy()
# score_normal = model.predict_proba(x_normal)[:, 1]
# randomlist = random.sample(range(0, len(score_normal)), total_anoamlies)
# x_anomalous = x_anomalous.to_numpy()
# x_normal = df_normal.to_numpy()
# x_anomalous = df_anomalous.to_numpy()
score_normal = pred(torch.tensor(x_normal, dtype=torch.float).to(dev), encoder, decoder)
max_score = max(score_normal)
# x_anomalous = np.loadtxt('age_hpw_anomaly.txt', dtype=int)
score_anomalous = pred(torch.tensor(x_anomalous.to_numpy(), dtype=torch.float).to(dev), encoder, decoder)
anom_indices = np.where(score_anomalous > max_score)[0].tolist()
randomlist = random.sample(range(0, len(score_normal)), len(anom_indices))
x_anomalous = x_anomalous.iloc[anom_indices]
x_anomalous.to_csv(anomaly_path + 'anomalous_data.csv', index=False)
# print(score_anomalous, score_normal)
sns.scatterplot(y=score_normal, x=list(range(len(score_normal))))
score_anomalous = pred(torch.tensor(x_anomalous.to_numpy(), dtype=torch.float).to(dev), encoder, decoder)
# print(len(score_anomalous), len(randomlist))
sns.scatterplot(y=score_anomalous, x=randomlist)
plt.savefig(anomaly_path + 'scatter_plot.png')
plt.show()
