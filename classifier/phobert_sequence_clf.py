import os
from transformers import RobertaConfig, BertModel
from transformers import RobertaForSequenceClassification


class PhoBertForSequenceClassification():
    def __init__(self):
        self.config = RobertaConfig.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'config.json')
        )

        self.config.num_labels = 3
        self.net = RobertaForSequenceClassification.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'model.bin'),
            config=self.config,
        )

    def __call__(self, x, attention_mask=None):
        loss, logits = self.net(x, attention_mask=attention_mask)
        return loss, logits


enc = PhoBertForSequenceClassification()
print(enc.net)