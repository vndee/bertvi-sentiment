import os
import torch
import torch.nn as nn


class SentimentAnalysisModel(torch.nn.Module):
    def __init__(self, encoder, feature_shape, num_classes):
        super(SentimentAnalysisModel, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.feature_shape = feature_shape

        self.linear_1 = nn.Linear(in_features=self.feature_shape, out_features=self.feature_shape//2)
        self.linear_2 = nn.Linear(in_features=self.feature_shape//2, out_features=self.num_classes)
        # self.soft_max = nn.Softmax(dim=1)
        self.linear = nn.Linear(5 * feature_shape, self.num_classes)

    def __call__(self, x, attention_mask=None):
        x = self.encoder(x, attention_mask, output_hidden_states=True, output_attentions=True)
        x = torch.cat((x[2][-1][:, 0, ...],
                       x[2][-2][:, 0, ...],
                       x[2][-3][:, 0, ...],
                       x[2][-4][:, 0, ...],
                       x[1]), -1)
        x = self.linear(x)
        # x = self.linear_1(x[1])
        # x = torch.nn.functional.relu(x)
        # x = self.linear_2(x)
        # x = self.soft_max(x)
        return x


