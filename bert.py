import os
import torch

from transformers import BertModel, BertTokenizer


class BertEncoder(torch.nn.Module):
    def __init__(self, pretrained_shorcut='bert-base-multilingual-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_shorcut)
        self.bert = BertModel.from_pretrained(pretrained_shorcut)

    def __call__(self, x):
        input_ids = torch.tensor([self.tokenizer.encode(x, add_special_tokens=True)])
        return self.bert(input_ids)
