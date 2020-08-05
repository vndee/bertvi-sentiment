import os
import torch.nn as nn
from transformers import ElectraConfig, ElectraModel


class ElectraEncoder(nn.Module):
    def __init__(self):
        super(ElectraEncoder, self).__init__()
        self.config = ElectraConfig().from_pretrained(
            os.path.join('pretrained', 'electra', 'config.json')
        )

        self.net = ElectraModel(self.config).from_pretrained(
            os.path.join('pretrained', 'electra', 'tf_model.h5')
        )

        print(self.net)


enc = ElectraEncoder()
