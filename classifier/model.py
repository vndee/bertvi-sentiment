import os
import torch
import torch.nn as nn

from models.flows import *
from models.nf_flow import NormalizingFlowModel
from torch.distributions import MultivariateNormal


class SentimentAnalysisModel(torch.nn.Module):
    def __init__(self, encoder, feature_shape, num_classes, num_flows=4, device='cpu'):
        super(SentimentAnalysisModel, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.feature_shape = feature_shape

        self.linear_1 = nn.Linear(in_features=self.feature_shape, out_features=self.feature_shape//2)
        self.linear_2 = nn.Linear(in_features=self.feature_shape//2, out_features=self.num_classes)
        # self.soft_max = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        self.flows = [NSF_CL(dim=4 * feature_shape) for _ in range(num_flows)]
        self.prior = MultivariateNormal(torch.zeros(4 * feature_shape, device=device),
                                        torch.eye(4 * feature_shape, device=device))
        self.nf_flows = NormalizingFlowModel(self.prior, self.flows)
        self.norm = nn.BatchNorm1d(4 * feature_shape)
        self.linear = nn.Linear(4 * feature_shape, self.num_classes)
        self.lstm_encoder = torch.nn.LSTM(768, 512, bidirectional=True, batch_first=True)
        self.linear_lstm = torch.nn.Linear(512, self.num_classes)

    def __call__(self, x, attention_mask=None):
        x = self.encoder(x, attention_mask, output_hidden_states=True, output_attentions=True)
        # o, c = self.lstm_encoder(x[2][-1])

        # x = torch.cat((x[2][-1][:, 0, ...],
        #                x[2][-2][:, 0, ...],
        #                x[2][-3][:, 0, ...],
        #                x[2][-4][:, 0, ...]), -1)
        # x = self.dropout(x)
        # z, prior_log_prob, log_det = self.nf_flows(x)
        # x = self.norm(x)
        # x = self.linear_lstm(x)
        x = self.linear_1(x[1])
        # x = torch.nn.functional.relu(x)
        x = self.linear_2(x)
        # x = self.soft_max(x)
        return x


