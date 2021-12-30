from torch import nn


class TorchEncoder(nn.Module):
    def __init__(self, in_dim=13, dropout=0.0, neg_slope=1e-2):
        super().__init__()

        # specify layer 1 - in 122, out 64
        self.encoder_L1 = nn.Linear(in_features=in_dim, out_features=8,
                                    bias=True)  # add linearity

        nn.init.xavier_uniform_(self.encoder_L1.weight)  # init weights according to [9]
        self.encoder_R1 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)  # add non-linearity according to [10]

        # specify layer 2 - in 64, out 32
        self.encoder_L2 = nn.Linear(8, 4, bias=True)
        nn.init.xavier_uniform_(self.encoder_L2.weight)
        self.encoder_R2 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)

        # specify layer 3 - in 32, out 16
        self.encoder_L3 = nn.Linear(4, 3, bias=True)
        nn.init.xavier_uniform_(self.encoder_L3.weight)
        self.encoder_R3 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)
        #
        # # specify layer 4 - in 16, out 8
        # self.encoder_L4 = nn.Linear(16, 8, bias=True)
        # nn.init.xavier_uniform_(self.encoder_L4.weight)
        # self.encoder_R4 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)
        #
        # # specify layer 5 - in 8, out 4
        # self.encoder_L5 = nn.Linear(8, 4, bias=True)
        # nn.init.xavier_uniform_(self.encoder_L5.weight)
        # self.encoder_R5 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)
        #
        # # specify layer 6 - in 4, out 3
        # self.encoder_L6 = nn.Linear(4, 3, bias=True)
        # nn.init.xavier_uniform_(self.encoder_L6.weight)
        # self.encoder_R6 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)

        # init dropout layer with probability p
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    # @torch.no_grad()
    def forward(self, data):
        # define forward pass through the network
        x = self.encoder_R1(self.dropout(self.encoder_L1(data)))
        x = self.encoder_R2(self.dropout(self.encoder_L2(x)))
        x = self.encoder_R3(self.encoder_L3(x))
        # x = self.encoder_R4(self.dropout(self.encoder_L4(x)))
        # x = self.encoder_R5(self.dropout(self.encoder_L5(x)))
        # x = self.encoder_R6(self.encoder_L6(x))  # don't apply dropout to the AE bottleneck

        return x


class TorchDecoder(nn.Module):

    def __init__(self, out_dim=122, dropout=0.0, neg_slope=1e-2):
        super(TorchDecoder, self).__init__()

        # specify layer 1 - in 3, out 4
        self.decoder_L1 = nn.Linear(in_features=3, out_features=4, bias=True)  # add linearity
        nn.init.xavier_uniform_(self.decoder_L1.weight)  # init weights according to [9]
        self.decoder_R1 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)  # add non-linearity according to [10]

        # specify layer 2 - in 4, out 8
        self.decoder_L2 = nn.Linear(4, 8, bias=True)
        nn.init.xavier_uniform_(self.decoder_L2.weight)
        self.decoder_R2 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)

        # specify layer 3 - in 8, out 16
        self.decoder_L3 = nn.Linear(8, out_dim, bias=True)
        nn.init.xavier_uniform_(self.decoder_L3.weight)
        self.decoder_R3 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)

        # specify layer 4 - in 16, out 32
        # self.decoder_L4 = nn.Linear(16, 32, bias=True)
        # nn.init.xavier_uniform_(self.decoder_L4.weight)
        # self.decoder_R4 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)
        #
        # # specify layer 5 - in 32, out 64
        # self.decoder_L5 = nn.Linear(32, 64, bias=True)
        # nn.init.xavier_uniform_(self.decoder_L5.weight)
        # self.decoder_R5 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)
        #
        # # specify layer 9 - in 64, out 122
        # self.decoder_L6 = nn.Linear(in_features=64, out_features=out_dim, bias=True)
        # nn.init.xavier_uniform_(self.decoder_L6.weight)
        # self.decoder_R6 = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)

        # init dropout layer with probability p
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    # @torch.no_grad()
    def forward(self, x):
        # define forward pass through the network
        x = self.decoder_R1(self.dropout(self.decoder_L1(x)))
        x = self.decoder_R2(self.dropout(self.decoder_L2(x)))
        x = self.decoder_R3(self.decoder_L3(x))
        # x = self.decoder_R4(self.dropout(self.decoder_L4(x)))
        # x = self.decoder_R5(self.dropout(self.decoder_L5(x)))
        # x = self.decoder_R6(self.decoder_L6(x))  # don't apply dropout to the AE output

        return x
