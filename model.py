import os
import torch
import torch.nn as nn


class SentimentAnalysisModel(torch.nn.Module):
    def __init__(self, encoder, feature_shape, num_classes):
        super(SentimentAnalysisModel, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.feature_shape = feature_shape

        self.linear_1 = nn.Linear(in_features=self.feature_shape, out_features=self.feature_shape/2)
        self.linear_2 = nn.Linear(in_features=self.feature_shape/2, out_features=self.num_classes)

    def __call__(self, x):
        x = self.encoder(x)
        x = self.linear_1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear_2(x)
        return x


