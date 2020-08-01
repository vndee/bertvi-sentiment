import os
import json
import torch

from transformers import RobertaModel, RobertaConfig


class PhoBertEncoder(torch.nn.Module):
    def __init__(self):
        super(PhoBertEncoder, self).__init__()

        self.config = RobertaConfig.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'config.json')
        )

        self.phobert = RobertaModel.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'model.bin'),
            config=self.config
        )

    def __call__(self, all_input_ids):
        features = self.phobert(all_input_ids)
        return features


