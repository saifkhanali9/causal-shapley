import pandas as pd
from datetime import datetime
import os
import torch.utils.data as data_utils
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch import autograd
from torch.utils.data import DataLoader

from autoencoder import TorchDecoder, TorchEncoder

# from scheer_module import DatasetPreperator

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
print(dev)

# convert encoded transactional data to torch Variable
dataset_name = 'census/'
model_path = '../output/model/' + dataset_name + 'all_epochs/'
os.makedirs(model_path, exist_ok=True)
x_train = pd.read_csv('../output/dataset/' + dataset_name + '/x_train.csv').to_numpy()
y_train = pd.read_csv('../output/dataset/' + dataset_name + '/y_train.csv').to_numpy()
n_features = x_train.shape[1]
torchX = torch.tensor(x_train, requires_grad=True, dtype=torch.float).to(dev)
# torchX = torch.from_numpy(x_train).float()
torchY = torch.tensor(y_train, dtype=torch.float).to(dev)
# torchX = torch.tensor(x_train).float()
# X = autograd.Variable(torchX)
train_data = data_utils.TensorDataset(torchX, torchY)

loss_function = nn.BCEWithLogitsLoss(reduction='mean')
learning_rate = 1e-3
num_epochs = 1000
mini_batch_size = 128

encoder_train = TorchEncoder(in_dim=n_features, dropout=0.25)
decoder_train = TorchDecoder(out_dim=n_features, dropout=0.25)
encoder_train.to(dev)
decoder_train.to(dev)
encoder_optimizer = torch.optim.Adam(encoder_train.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder_train.parameters(), lr=learning_rate)

dataloader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)
# dataloader = DataLoader(torchX, batch_size=mini_batch_size, shuffle=True)
os.makedirs(model_path, exist_ok=True)
iterations_performed = 0
# train autoencoder model
for epoch in range(num_epochs):
    # encoder_train.to(dev)
    # decoder_train.to(dev)
    encoder_train.train()
    decoder_train.train()
    mini_batch_count = 0
    start_time = datetime.now()
    for i, (mini_batch_data, y) in enumerate(dataloader):
        iterations_performed += 1

        # increase mini batch counter
        mini_batch_count += 1

        # convert mini batch to torch variable
        # mini_batch_data.requires_grad = True

        # =================== (1) forward pass ===================================

        # run forward pass
        z_representation = encoder_train(mini_batch_data)  # encode mini-batch data
        mini_batch_reconstruction = decoder_train(z_representation)  # decode mini-batch data

        # =================== (2) compute reconstruction loss ====================

        # determine reconstruction loss
        reconstruction_loss = loss_function(mini_batch_reconstruction, mini_batch_data)
        # reconstruction_loss.requires_grad = True
        # =================== (3) backward pass ==================================

        # reset graph gradients
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()

        # run backward pass
        reconstruction_loss.backward()

        # =================== (4) update model parameters ========================

        # update network parameters
        decoder_optimizer.step()
        encoder_optimizer.step()

        # =================== monitor training progress ==========================

        # print training progress each 1'000 mini-batches
        # if mini_batch_count % 100 == 0:
        #     # print('\n\n\n\n\n\n\n\n========================================')
        #     # for params in encoder_train.parameters():
        #     #     print(params.data)
        #     #     break
        #
        #     now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
        #     end_time = datetime.now() - start_time
        #     remaining_iters = (len(x_train) / mini_batch_size * num_epochs) - iterations_performed
        #     # temp1 = (len(x_train) / mini_batch_count * num_epochs)
        #     # temp = ((i + 1) * (epoch + 1))
        #     eta = str(end_time * remaining_iters / 60).split(".")[0]
        #     print(
        #         '[LOG {}] training status, epoch: [{:04}/{:04}], batch: {:04}, loss: {}, mode: {}, time required: {}, ETA: {}'.format(
        #             now, (epoch + 1), num_epochs, mini_batch_count, np.round(reconstruction_loss.item(), 4), dev,
        #             end_time, eta))
        #
        #     # reset timer
        #     start_time = datetime.now()

        # =================== evaluate model performance =============================

        # set networks in evaluation mode (don't apply dropout)
    encoder_train.eval()
    decoder_train.eval()

    # reconstruct encoded transactional data
    reconstruction = decoder_train(encoder_train(torchX))

    # determine reconstruction loss - all transactions
    reconstruction_loss_all = loss_function(reconstruction, torchX)

    # print reconstuction loss results
    now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
    print('[LOG {}] training status, epoch: [{:04}/{:04}], loss: {:.10f}'.format(now, (epoch + 1), num_epochs,
                                                                                 reconstruction_loss_all.item()))

    # =================== save model snapshot to disk ============================

    # save trained encoder model file to disk
    encoder_model_name = "ep_{}_encoder_model.pth".format((epoch + 1))
    torch.save(encoder_train.state_dict(), os.path.join(model_path, encoder_model_name))

    # save trained decoder model file to disk
    decoder_model_name = "ep_{}_decoder_model.pth".format((epoch + 1))
    torch.save(decoder_train.state_dict(), os.path.join(model_path, decoder_model_name))