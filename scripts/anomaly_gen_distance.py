import pandas as pd
import numpy as np
import torch
from joblib import load
import torch.nn as nn
import seaborn as sns
import random
import shutil
import os
from autoencoder import TorchEncoder, TorchDecoder
from matplotlib import pyplot as plt

if torch.cuda.is_available():
    #     dev = "cuda:0"
    dev = "cpu"
else:
    dev = "cpu"


def pred(x, enc, dec):
    # x = x[1]
    y = enc(x)
    y = dec(y)
    loss_fn = nn.MSELoss(reduction='none')
    los_val = loss_fn(y, x)
    return torch.mean(los_val, dim=1).cpu().detach().numpy()


total_anoamlies = 130
epoch = 1000
dataset_name = 'census2'
model_path = f'../output/model/{dataset_name}/all_epochs_no_dropouts/'
encoder_model = model_path + 'ep_' + str(epoch) + '_encoder_model.pth'
decoder_model = model_path + 'ep_' + str(epoch) + '_decoder_model.pth'
df_normal = pd.read_csv(f'../output/dataset/{dataset_name}/x_test.csv')
df_anomalous = pd.read_csv(f'../output/dataset/{dataset_name}/x_test.csv')
anomaly_description = 'Features sex=19, age=80, education=1 were set to give most distance from their neighbours.'
df_anomalous['age'] = 80
df_anomalous['sex'] = 19
df_anomalous['education'] = 1
# df_anomalous['hours_per_week'] = 1
# df_anomalous['education'] = 16
# x_anomalous = df_anomalous.sample(n=total_anoamlies)

# Started writing
# if False:
dataset_path = f'../output/anomaly_included/{dataset_name}/'
anomaly_type = 'distance'
anomaly_name = '_age_education_sex/'
anomaly_path = dataset_path + anomaly_type + anomaly_name
os.makedirs(anomaly_path, exist_ok=True)
with open(anomaly_path + 'readme.txt', 'w') as f:
    f.write(anomaly_description)

# Train set
shutil.copyfile('../output/dataset/census3/x_train.csv', anomaly_path + '/x_train.csv')

# Antoine's set
df_normal[:1000].to_csv(anomaly_path + 'x_test_antoine.csv', index=False)
# Test set
df_normal[1000:].to_csv(anomaly_path + 'x_test.csv', index=False)
# Anomalies Antoine
df_anomalous[:1000].to_csv(anomaly_path + 'anomalous_data_antoine.csv', index=False)
# Anomalies
df_anomalous[1000:].to_csv(anomaly_path + 'anomalous_data.csv', index=False)

x_anomalous = df_anomalous.to_numpy()
x_normal = df_normal.to_numpy()

encoder = TorchEncoder(in_dim=x_normal.shape[1]).to(dev)
decoder = TorchDecoder(out_dim=x_normal.shape[1]).to(dev)
encoder.load_state_dict(torch.load(encoder_model))
decoder.load_state_dict(torch.load(decoder_model))
# x_anomalous = df_anomalous.to_numpy()
score_normal = pred(torch.tensor(x_normal, dtype=torch.float).to(dev), encoder, decoder)
randomlist = random.sample(range(0, len(score_normal)), total_anoamlies)
# x_anomalous = np.loadtxt('age_hpw_anomaly.txt', dtype=int)
score_anomalous = pred(torch.tensor(x_anomalous, dtype=torch.float).to(dev), encoder, decoder)
print(score_anomalous, score_normal)
sns.scatterplot(y=score_normal, x=list(range(len(score_normal))))
sns.scatterplot(y=score_anomalous, x=list(range(len(score_anomalous))))
plt.savefig(anomaly_path + '/scatter_plot.png')
plt.show()
# print(model.predict_proba(x_normal))
# print(model.predict_proba(np.zeros(len(x_normal[0])).reshape(1,-1)))
