import os
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
from vncorenlp import VnCoreNLP
from torch.autograd import Variable
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader


class BiLSTM_Attention(nn.Module):
    def __init__(self, enc=None, embedding_size=768, lstm_hidden_size=512, num_classes=3, device='cuda'):
        super(BiLSTM_Attention, self).__init__()
        self.encoder = enc
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_classes = num_classes
        self.device = device

        self.lstm = torch.nn.LSTM(self.embedding_size, self.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.out = torch.nn.Linear(self.lstm_hidden_size * 2, self.num_classes)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.lstm_hidden_size * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = torch.nn.functional.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data

    def forward(self, x, attention_mask):
        x = self.encoder(x, attention_mask=attention_mask)[0]
        hidden_state = Variable(torch.zeros(1 * 2, x.shape[0], self.lstm_hidden_size, device=self.device))
        cell_state = Variable(torch.zeros(1 * 2, x.shape[0], self.lstm_hidden_size, device=self.device))

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (hidden_state, cell_state))
        attn_output, attention = self.attention_net(output, final_hidden_state)
        x = self.out(attn_output)
        return x
