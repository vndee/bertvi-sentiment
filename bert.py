import os
import torch

from transformers import BertModel, BertTokenizer


class BertEncoder(torch.nn.Module):
    def __init__(self, pretrained_shortcut='bert-base-multilingual-cased'):
        super(BertEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_shortcut)
        self.bert = BertModel.from_pretrained(pretrained_shortcut)

    def __call__(self, x, attention_mask=None, output_hidden_states=None, output_attentions=None):
        return self.bert(x,
                         attention_mask=attention_mask,
                         output_hidden_states=output_hidden_states,
                         output_attentions=output_attentions)
