import os
import json
import torch

from transformers import RobertaModel, RobertaConfig
from transformers import RobertaForSequenceClassification


class PhoBertEncoder(torch.nn.Module):
    def __init__(self):
        super(PhoBertEncoder, self).__init__()
        self.config = RobertaConfig.from_pretrained(
            os.path.join(os.getcwd(), '../pretrained', 'PhoBERT_base_transformers', 'config.json')
        )

        self.phobert = RobertaModel.from_pretrained(
            os.path.join(os.getcwd(), '../pretrained', 'PhoBERT_base_transformers', 'models.bin'),
            config=self.config,
        )

    def __call__(self, all_input_ids, attention_mask=None, output_hidden_states=None, output_attentions=None):
        features = self.phobert(all_input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=output_hidden_states,
                                output_attentions=output_attentions)
        return features
