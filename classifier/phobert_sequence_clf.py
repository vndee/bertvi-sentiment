import os
from transformers import RobertaConfig
from transformers import RobertaForSequenceClassification


class PhoBertForSequenceClassification:
    def __init__(self):
        self.config = RobertaConfig.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'config.json')
        )
        self.net = RobertaForSequenceClassification.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'model.bin'),
            config=self.config
        )


enc = PhoBertForSequenceClassification()